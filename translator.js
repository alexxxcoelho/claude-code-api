import { v4 as uuidv4 } from 'uuid';

/**
 * Convert OpenAI chat messages to Claude Code stream-json input format
 */
export function openaiToClaudeInput(messages, sessionId) {
  // Extract system prompt if present
  const systemMessages = messages.filter(m => m.role === 'system');
  const otherMessages = messages.filter(m => m.role !== 'system');

  // Combine all messages into a single prompt
  // Claude Code manages its own conversation history via session
  let prompt = '';

  if (systemMessages.length > 0) {
    prompt += systemMessages.map(m => m.content).join('\n') + '\n\n';
  }

  // Get the latest user message (Claude Code maintains history internally)
  const lastUserMessage = otherMessages.filter(m => m.role === 'user').pop();
  if (lastUserMessage) {
    prompt += lastUserMessage.content;
  }

  return {
    type: 'user',
    uuid: uuidv4(),
    session_id: sessionId,
    message: {
      role: 'user',
      content: [{ type: 'text', text: prompt }]
    },
    parent_tool_use_id: null
  };
}

/**
 * Convert Claude Code stream event to OpenAI SSE chunk format
 */
export function claudeToOpenaiChunk(claudeEvent, completionId, model) {
  const chunk = {
    id: completionId,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      delta: {},
      finish_reason: null
    }]
  };

  // Handle different Claude event types
  if (claudeEvent.type === 'stream_event') {
    const event = claudeEvent.event;

    if (event?.type === 'content_block_delta') {
      if (event.delta?.type === 'text_delta' && event.delta?.text) {
        chunk.choices[0].delta.content = event.delta.text;
      }
    } else if (event?.type === 'message_start') {
      chunk.choices[0].delta.role = 'assistant';
    }
  } else if (claudeEvent.type === 'assistant') {
    // Full assistant message - extract text content
    const content = claudeEvent.message?.content;
    if (Array.isArray(content)) {
      const textContent = content
        .filter(c => c.type === 'text')
        .map(c => c.text)
        .join('');
      if (textContent) {
        chunk.choices[0].delta.content = textContent;
      }
    }
  } else if (claudeEvent.type === 'message_stop' || claudeEvent.type === 'result') {
    chunk.choices[0].delta = {};
    chunk.choices[0].finish_reason = 'stop';
  }

  return chunk;
}

/**
 * Build a system-prompt section that exposes OpenAI-style function/tool
 * definitions to the model and instructs it to emit calls as strict JSON. This
 * is a prompt-based shim: `claude -p` has no native client-defined tool calling,
 * so we ask for a JSON envelope and parse it back with parseAssistantToolCalls.
 */
export function toolsToSystemPrompt(tools, toolChoice) {
  if (!Array.isArray(tools) || tools.length === 0) return '';
  const fns = tools
    .filter(t => t?.type === 'function' && t.function)
    .map(t => t.function);
  if (fns.length === 0) return '';

  if (toolChoice === 'none') return '';

  const list = fns
    .map(
      f =>
        `- ${f.name}: ${f.description || ''}\n  parameters (JSON Schema): ${JSON.stringify(
          f.parameters || {}
        )}`
    )
    .join('\n');

  let forced = '';
  if (toolChoice && typeof toolChoice === 'object' && toolChoice.function?.name) {
    forced = `\nYou MUST call the function "${toolChoice.function.name}".`;
  } else if (toolChoice === 'required') {
    forced = '\nYou MUST call one of the functions.';
  }

  return [
    '',
    '',
    '# Function calling',
    'You can call the following functions:',
    list,
    'To call one or more functions, respond with ONLY a JSON object, no prose and no code fences:',
    '{"tool_calls":[{"name":"<function name>","arguments":{<args matching the schema>}}]}',
    'Use the exact argument names from the schema. If you do not need a function, answer normally in plain text.' +
      forced
  ].join('\n');
}

/**
 * Parse a model text response into OpenAI tool_calls, or null if it isn't a
 * tool-call envelope. Tolerates code fences and surrounding prose.
 */
export function parseAssistantToolCalls(text) {
  if (!text) return null;
  let s = String(text).trim();

  // Strip ```json ... ``` fences if present
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fence) s = fence[1].trim();

  let obj = null;
  try {
    obj = JSON.parse(s);
  } catch (e) {
    const m = s.match(/\{[\s\S]*\}/);
    if (m) {
      try {
        obj = JSON.parse(m[0]);
      } catch (e2) {
        return null;
      }
    }
  }
  if (!obj || typeof obj !== 'object') return null;

  let raw = null;
  if (Array.isArray(obj.tool_calls)) raw = obj.tool_calls;
  else if (obj.tool_call && typeof obj.tool_call === 'object') raw = [obj.tool_call];
  if (!raw || raw.length === 0) return null;

  return raw.map(c => {
    const name = c.name || c.function?.name || 'unknown';
    let args = c.arguments ?? c.function?.arguments ?? {};
    if (typeof args !== 'string') args = JSON.stringify(args);
    return {
      id: `call_${uuidv4().replace(/-/g, '').slice(0, 24)}`,
      type: 'function',
      function: { name, arguments: args }
    };
  });
}

/**
 * Convert Claude Code result to OpenAI non-streaming response.
 * If options.tools is set and the output is a tool-call envelope, returns a
 * tool_calls message with finish_reason 'tool_calls'.
 */
export function claudeResultToOpenai(resultEvent, assistantMessages, completionId, model, options = {}) {
  // Collect all text from assistant messages
  let fullContent = '';

  for (const msg of assistantMessages) {
    if (msg.type === 'assistant' && msg.message?.content) {
      const textParts = msg.message.content
        .filter(c => c.type === 'text')
        .map(c => c.text);
      fullContent += textParts.join('');
    }
  }

  // If we have a result event with result text, use that
  if (resultEvent?.result) {
    fullContent = resultEvent.result;
  }

  const message = { role: 'assistant', content: fullContent };
  let finishReason = 'stop';

  if (options.tools) {
    const toolCalls = parseAssistantToolCalls(fullContent);
    if (toolCalls) {
      message.content = null;
      message.tool_calls = toolCalls;
      finishReason = 'tool_calls';
    }
  }

  return {
    id: completionId,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: {
      prompt_tokens: resultEvent?.usage?.input_tokens || 0,
      completion_tokens: resultEvent?.usage?.output_tokens || 0,
      total_tokens: (resultEvent?.usage?.input_tokens || 0) + (resultEvent?.usage?.output_tokens || 0)
    }
  };
}

/**
 * Format an OpenAI SSE chunk for transmission
 */
export function formatSSE(chunk) {
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/**
 * Format the final SSE done message
 */
export function formatSSEDone() {
  return 'data: [DONE]\n\n';
}

/**
 * Parse a line of NDJSON from Claude Code output
 */
export function parseClaudeLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return null;

  try {
    return JSON.parse(trimmed);
  } catch (e) {
    // Not valid JSON, might be partial or debug output
    return null;
  }
}
