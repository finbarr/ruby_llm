# frozen_string_literal: true

module RubyLLM
  # Base class for LLM providers.
  class Provider
    include Streaming

    attr_reader :config, :connection, :hooks

    def initialize(config)
      @config = config
      @hooks = {
        before_request: nil,
        after_response: nil,
        on_error: nil,
        on_retry: nil
      }
      ensure_configured!
      @connection = Connection.new(self, @config)
    end

    def on_before_request(&block)
      @hooks[:before_request] = block
      self
    end

    def on_after_response(&block)
      @hooks[:after_response] = block
      self
    end

    def on_error(&block)
      @hooks[:on_error] = block
      self
    end

    def on_retry(&block)
      @hooks[:on_retry] = block
      self
    end

    def api_base
      raise NotImplementedError
    end

    def headers
      {}
    end

    def slug
      self.class.slug
    end

    def name
      self.class.name
    end

    def capabilities
      self.class.capabilities
    end

    def configuration_requirements
      self.class.configuration_requirements
    end

    def complete(messages, tools:, temperature:, model:, params: {}, headers: {}, schema: nil, &) # rubocop:disable Metrics/ParameterLists
      normalized_temperature = maybe_normalize_temperature(temperature, model)

      payload = Utils.deep_merge(
        params,
        render_payload(
          messages,
          tools: tools,
          temperature: normalized_temperature,
          model: model,
          stream: block_given?,
          schema: schema
        )
      )

      # Convert payload to raw JSON for the before_request hook
      raw_request_json = JSON.generate(payload, ascii_only: false)

      # Call before_request hook with ONLY raw JSON
      @hooks[:before_request]&.call(raw_request_json)

      begin
        response = if block_given?
                     stream_response @connection, payload, headers, &
                   else
                     sync_response @connection, payload, headers
                   end

        trigger_after_response_hook(response)
        response
      rescue StandardError => e
        trigger_error_hook(e)
        raise
      end
    end

    def list_models
      response = @connection.get models_url
      parse_list_models_response response, slug, capabilities
    end

    def embed(text, model:, dimensions:)
      payload = render_embedding_payload(text, model:, dimensions:)
      response = @connection.post(embedding_url(model:), payload)
      parse_embedding_response(response, model:, text:)
    end

    def paint(prompt, model:, size:)
      payload = render_image_payload(prompt, model:, size:)
      response = @connection.post images_url, payload
      parse_image_response(response, model:)
    end

    def configured?
      configuration_requirements.all? { |req| @config.send(req) }
    end

    def local?
      self.class.local?
    end

    def remote?
      self.class.remote?
    end

    def parse_error(response)
      return if response.body.empty?

      body = try_parse_json(response.body)
      case body
      when Hash
        body.dig('error', 'message')
      when Array
        body.map do |part|
          part.dig('error', 'message')
        end.join('. ')
      else
        body
      end
    end

    def format_messages(messages)
      messages.map do |msg|
        {
          role: msg.role.to_s,
          content: msg.content
        }
      end
    end

    def format_tool_calls(_tool_calls)
      nil
    end

    def parse_tool_calls(_tool_calls)
      nil
    end

    class << self
      def name
        to_s.split('::').last
      end

      def slug
        name.downcase
      end

      def capabilities
        raise NotImplementedError
      end

      def configuration_requirements
        []
      end

      def local?
        false
      end

      def remote?
        !local?
      end

      def configured?(config)
        configuration_requirements.all? { |req| config.send(req) }
      end

      def register(name, provider_class)
        providers[name.to_sym] = provider_class
      end

      def for(model)
        model_info = Models.find(model)
        providers[model_info.provider.to_sym]
      end

      def providers
        @providers ||= {}
      end

      def local_providers
        providers.select { |_slug, provider_class| provider_class.local? }
      end

      def remote_providers
        providers.select { |_slug, provider_class| provider_class.remote? }
      end

      def configured_providers(config)
        providers.select do |_slug, provider_class|
          provider_class.configured?(config)
        end.values
      end

      def configured_remote_providers(config)
        providers.select do |_slug, provider_class|
          provider_class.remote? && provider_class.configured?(config)
        end.values
      end
    end

    private

    def try_parse_json(maybe_json)
      return maybe_json unless maybe_json.is_a?(String)

      JSON.parse(maybe_json)
    rescue JSON::ParserError
      maybe_json
    end

    def ensure_configured!
      missing = configuration_requirements.reject { |req| @config.send(req) }
      return if missing.empty?

      raise ConfigurationError, "Missing configuration for #{name}: #{missing.join(', ')}"
    end

    def maybe_normalize_temperature(temperature, _model_id)
      temperature
    end

    def sync_response(connection, payload, additional_headers = {})
      response = connection.post completion_url, payload do |req|
        req.headers = additional_headers.merge(req.headers) unless additional_headers.empty?
      end
      parsed_message = parse_completion_response response
      # Attach raw request and response to the message for hooks
      if response.respond_to?(:raw_request)
        parsed_message.define_singleton_method(:raw_request) do
          response.raw_request
        end
      end
      if response.respond_to?(:raw_response)
        parsed_message.define_singleton_method(:raw_response) do
          response.raw_response
        end
      end
      parsed_message
    end

    def trigger_after_response_hook(message)
      return unless @hooks[:after_response]

      # Get raw response JSON from the message object if available
      raw_response_json = if message.respond_to?(:raw_response)
                            message.raw_response
                          elsif message.respond_to?(:raw) && message.raw&.body
                            # For Anthropic, the raw is the Faraday response, body is already parsed
                            JSON.generate(message.raw.body, ascii_only: false)
                          end

      # Call after_response hook with ONLY raw JSON
      @hooks[:after_response].call(raw_response_json)
    end

    def trigger_error_hook(error)
      return unless @hooks[:on_error]

      @hooks[:on_error].call(error)
    end

    def tool_to_hash(tool)
      {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters.transform_values do |p|
          { type: p.type, description: p.description, required: p.required }
        end
      }
    end

    def message_to_hash(message)
      {
        role: message.role,
        content: message.content,
        tool_calls: message.tool_calls&.transform_values do |tc|
          { id: tc.id, name: tc.name, arguments: tc.arguments }
        end,
        input_tokens: message.input_tokens,
        output_tokens: message.output_tokens,
        model_id: message.model_id
      }
    end
  end
end
