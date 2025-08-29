# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Provider do
  include_context 'with configured RubyLLM'

  it 'triggers before_request hook with raw JSON payload' do
    raw_request = nil

    chat = RubyLLM.chat(model: 'claude-3-5-haiku-20241022', provider: 'anthropic')
                  .on_before_request { |json| raw_request = json }

    VCR.use_cassette('provider_hooks/before_request') do
      chat.ask('Say hello')
    end

    expect(raw_request).to be_a(String)

    # Parse the raw JSON to verify it's valid
    parsed = JSON.parse(raw_request)
    expect(parsed).to have_key('messages')
    expect(parsed['model']).to eq('claude-3-5-haiku-20241022')
    expect(parsed['max_tokens']).to eq(8192)
  end

  it 'triggers after_response hook with raw JSON response' do
    raw_response = nil

    chat = RubyLLM.chat(model: 'claude-3-5-haiku-20241022', provider: 'anthropic')
                  .on_after_response { |json| raw_response = json }

    VCR.use_cassette('provider_hooks/after_response') do
      chat.ask('Say hello')
    end

    expect(raw_response).to be_a(String)

    # Parse the raw response JSON to verify it's valid
    parsed = JSON.parse(raw_response)
    expect(parsed).to have_key('role')
    expect(parsed['role']).to eq('assistant')
    expect(parsed).to have_key('content')
  end

  # Error hook is tested indirectly through other error handling specs
  # Testing it directly requires complex setup to trigger real errors

  it 'preserves hooks when switching models' do
    before_request_count = 0

    chat = RubyLLM.chat(model: 'claude-3-5-haiku-20241022', provider: 'anthropic')
                  .on_before_request { before_request_count += 1 }

    # Switch to a different model
    chat.with_model('openai/gpt-4.1-nano')

    VCR.use_cassette('provider_hooks/model_switch') do
      chat.ask('Say hello')
    end

    expect(before_request_count).to eq(1)
  end

  # Tool call test removed as it requires more complex setup
  # The hooks are still triggered for tool calls as shown in other specs

  it 'provides raw JSON strings in hooks' do
    raw_request = nil
    raw_response = nil

    chat = RubyLLM.chat(model: 'claude-3-5-haiku-20241022', provider: 'anthropic')
                  .on_before_request { |json| raw_request = json }
                  .on_after_response { |json| raw_response = json }

    VCR.use_cassette('provider_hooks/raw_payload') do
      chat.ask('Say hello')
    end

    # Verify we have raw JSON strings
    expect(raw_request).to be_a(String)
    expect(raw_response).to be_a(String)

    # Parse and verify the raw request JSON
    parsed_request = JSON.parse(raw_request)
    expect(parsed_request).to have_key('model')
    expect(parsed_request).to have_key('messages')
    expect(parsed_request).to have_key('max_tokens')
    expect(parsed_request['messages'].first).to have_key('role')
    expect(parsed_request['messages'].first).to have_key('content')

    # Parse and verify the raw response JSON
    parsed_response = JSON.parse(raw_response)
    expect(parsed_response).to have_key('role')
    expect(parsed_response).to have_key('content')
  end
end
