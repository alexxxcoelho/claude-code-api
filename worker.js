import { spawn } from 'child_process'
import { createInterface } from 'readline'
import { EventEmitter } from 'events'
import { v4 as uuidv4 } from 'uuid'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import {
  claudeToOpenaiChunk,
  claudeResultToOpenai,
  parseClaudeLine,
  toolsToSystemPrompt,
  parseAssistantToolCalls
} from './translator.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Use local Claude installation instead of global
const CLAUDE_CLI_PATH = join(
  __dirname,
  'node_modules',
  '@anthropic-ai',
  'claude-code',
  'cli.js'
)

// Check if OAuth mode is enabled
const USE_OAUTH = process.env.AUTH_MODE === 'oauth'

// Optional model alias map (JSON). Maps the model id a client sends to the value
// passed to `claude --model`, e.g.
//   MODEL_ALIASES='{"anthropic-sonnet-4-6":"claude-sonnet-4-6","gpt-4o":"opus"}'
// Unmapped models are passed through as-is (so "sonnet", "opus", "haiku" or a
// full "claude-..." id work directly). The sentinel "claude-code" and an empty
// model fall back to the CLI's default model.
let MODEL_ALIASES = {}
try {
  if (process.env.MODEL_ALIASES) {
    MODEL_ALIASES = JSON.parse(process.env.MODEL_ALIASES)
  }
} catch (err) {
  console.error('[Worker] Invalid MODEL_ALIASES JSON, ignoring:', err.message)
}

export function resolveModel (model) {
  if (!model) return null
  // Strip the optional ":<conversationId>" routing suffix.
  const base = String(model).split(':')[0].trim()
  if (!base || base === 'claude-code') return null // use the CLI default model
  return MODEL_ALIASES[base] || base
}

// Optional default system prompt used when the client doesn't send a system
// message. Overriding the system prompt (below) strips Claude Code's coding-agent
// framing so the proxy behaves like a plain chat model.
const DEFAULT_SYSTEM_PROMPT =
  process.env.CLAUDE_SYSTEM_PROMPT || 'You are a helpful assistant.'

/**
 * Extract plain text from an OpenAI `content` field, which may be a string or an
 * array of parts (e.g. [{type:'text',text:'...'}]). Without this, an array
 * content is coerced to "[object Object]" and reaches the model as garbage.
 */
export function extractText (content) {
  if (content == null) return ''
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map(part =>
        typeof part === 'string' ? part : part?.text ?? part?.content ?? ''
      )
      .filter(Boolean)
      .join('')
  }
  return String(content)
}

/**
 * Turn OpenAI chat messages into { systemPrompt, prompt }.
 * - system messages become the overridden system prompt (chat behaviour).
 * - a single user turn becomes the prompt verbatim; multi-turn conversations are
 *   rendered as a Human/Assistant transcript so prior context is preserved even
 *   when the client doesn't reuse a conversation id.
 */
export function messagesToPrompt (messages) {
  const systemPrompt = messages
    .filter(m => m.role === 'system')
    .map(m => extractText(m.content))
    .filter(Boolean)
    .join('\n\n')

  const convo = messages.filter(
    m => m.role === 'user' || m.role === 'assistant' || m.role === 'tool'
  )

  const render = m => {
    if (m.role === 'assistant') {
      const text = extractText(m.content)
      if (Array.isArray(m.tool_calls) && m.tool_calls.length) {
        const calls = m.tool_calls
          .map(tc => `${tc.function?.name}(${tc.function?.arguments})`)
          .join(', ')
        return `Assistant: ${text ? text + ' ' : ''}[called: ${calls}]`
      }
      return `Assistant: ${text}`
    }
    if (m.role === 'tool') {
      return `Tool result (${m.tool_call_id || ''}): ${extractText(m.content)}`
    }
    return `Human: ${extractText(m.content)}`
  }

  let prompt
  if (convo.length <= 1) {
    prompt = convo.length ? extractText(convo[0].content) : ''
  } else {
    prompt = convo.map(render).join('\n\n')
  }
  return { systemPrompt, prompt }
}

/**
 * Manages a single Claude Code CLI process
 */
export class ClaudeWorker extends EventEmitter {
  constructor (conversationId, options = {}) {
    super()
    this.conversationId = conversationId
    this.workspaceDir = options.workspaceDir || '/workspace'
    this.proc = null
    this.readline = null
    this.lastUsed = Date.now()
    this.ready = false
    this.pendingRequests = new Map() // requestId -> { resolve, reject, chunks, resultEvent }
  }

  /**
   * Build the `claude` argument list shared by streaming and non-streaming.
   *
   * Security: tool access and permission bypass are OPT-IN, not the default.
   *   CLAUDE_TOOLS  -> value passed to `--tools` ("" disables ALL tools, the
   *                    safe default; "default" enables all; or a list like
   *                    "Read,Edit"). With tools disabled the worker can only
   *                    generate text, removing the RCE surface for an exposed
   *                    HTTP endpoint.
   *   CLAUDE_SKIP_PERMISSIONS=true -> restores --dangerously-skip-permissions.
   *
   * NOTE: `--tools` is variadic, so it is always followed by another flag and
   * never sits immediately before the prompt, which would otherwise be swallowed
   * as a tool name. The prompt is always the final positional argument.
   */
  buildClaudeArgs (prompt, model, { stream = false, systemPrompt = '' } = {}) {
    const toolsSetting = process.env.CLAUDE_TOOLS ?? ''
    const skipPermissions = process.env.CLAUDE_SKIP_PERMISSIONS === 'true'

    const args = ['-p', '--tools', toolsSetting]
    // Override the system prompt so the proxy behaves as a chat model rather
    // than the Claude Code coding agent, and drop the dynamic env/skills/memory
    // sections. Client system message wins; otherwise a neutral default.
    args.push(
      '--system-prompt',
      systemPrompt || DEFAULT_SYSTEM_PROMPT,
      '--exclude-dynamic-system-prompt-sections'
    )
    if (stream) {
      // Realtime token deltas as NDJSON; --verbose is required with -p.
      args.push(
        '--output-format',
        'stream-json',
        '--verbose',
        '--include-partial-messages'
      )
    } else {
      args.push('--output-format', 'json')
    }
    args.push('--session-id', this.conversationId)

    const resolvedModel = resolveModel(model)
    if (resolvedModel) {
      args.push('--model', resolvedModel)
    }
    if (skipPermissions) {
      args.push('--dangerously-skip-permissions')
    }
    args.push(prompt) // prompt MUST be the final argument
    return args
  }

  /**
   * Resolve the command + args for the current auth mode: the global `claude`
   * binary in oauth mode, or the bundled CLI run via node in API-key mode.
   */
  resolveSpawnCommand (args) {
    const command = USE_OAUTH
      ? process.platform === 'win32'
        ? 'claude.cmd'
        : 'claude'
      : 'node'
    const finalArgs = USE_OAUTH ? args : [CLAUDE_CLI_PATH, ...args]
    return { command, finalArgs }
  }

  /**
   * Spawn the Claude Code CLI process
   * Note: We spawn a new process for each request in non-streaming mode
   */
  async spawn (prompt, model, systemPrompt) {
    if (this.proc && this.proc.exitCode === null) {
      // Kill existing process if still running
      this.proc.kill()
    }

    // Non-streaming: one JSON object on stdout (parsed in handleJsonOutput).
    const args = this.buildClaudeArgs(prompt, model, { stream: false, systemPrompt })
    const { command, finalArgs } = this.resolveSpawnCommand(args)

    console.log(
      `[Worker ${this.conversationId}] ${
        USE_OAUTH ? 'OAuth' : 'API Key'
      } mode: ${command} -p "${prompt.substring(0, 30)}..."`
    )

    this.proc = spawn(command, finalArgs, {
      cwd: this.workspaceDir,
      env: {
        ...process.env,
        // Ensure no TTY interactions
        TERM: 'dumb',
        NO_COLOR: '1'
      },
      stdio: ['ignore', 'pipe', 'pipe'], // ignore stdin since we pass prompt via args
      shell: USE_OAUTH && process.platform === 'win32', // Shell only needed for .cmd files
      windowsHide: true
    })

    // Collect all stdout data
    let stdoutData = ''
    this.proc.stdout.on('data', data => {
      stdoutData += data.toString()
    })

    this.proc.stdout.on('end', () => {
      this.handleJsonOutput(stdoutData)
    })

    // Handle stderr (for debugging)
    this.proc.stderr.on('data', data => {
      const msg = data.toString().trim()
      if (msg) {
        console.error(`[Worker ${this.conversationId}] stderr: ${msg}`)
      }
    })

    // Handle process exit
    this.proc.on('close', code => {
      console.log(
        `[Worker ${this.conversationId}] Process exited with code ${code}`
      )
      this.proc = null
      this.ready = false
      this.emit('close', code)

      // Reject any pending requests if no output was received
      for (const [reqId, pending] of this.pendingRequests) {
        if (!pending.resolved) {
          pending.reject(
            new Error(
              `Worker process exited with code ${code} before completing`
            )
          )
        }
      }
      this.pendingRequests.clear()
    })

    this.proc.on('error', err => {
      console.error(`[Worker ${this.conversationId}] Process error:`, err)
      this.emit('error', err)
    })

    this.ready = true
    console.log(`[Worker ${this.conversationId}] Spawned successfully`)
  }

  /**
   * Handle complete JSON output from Claude (non-streaming mode)
   */
  handleJsonOutput (jsonStr) {
    try {
      if (!jsonStr || jsonStr.trim() === '') {
        console.error(`[Worker ${this.conversationId}] Empty output received`)
        const requestId = this.getLatestRequestId()
        if (requestId && this.pendingRequests.has(requestId)) {
          this.pendingRequests
            .get(requestId)
            .reject(new Error('Empty output from Claude CLI'))
          this.pendingRequests.delete(requestId)
        }
        return
      }

      const result = JSON.parse(jsonStr)

      // Find the pending request (should only be one in non-streaming mode)
      const requestId = this.getLatestRequestId()

      if (requestId && this.pendingRequests.has(requestId)) {
        const pending = this.pendingRequests.get(requestId)
        pending.resolved = true // Mark as resolved so close handler doesn't reject
        pending.resolve({
          chunks: [],
          result: result
        })
        this.pendingRequests.delete(requestId)
      } else {
        console.error(
          `[Worker ${this.conversationId}] No pending request for result`
        )
      }
    } catch (err) {
      console.error(
        `[Worker ${this.conversationId}] Failed to parse JSON output:`,
        err
      )
      console.error(`[Worker ${this.conversationId}] Raw output:`, jsonStr)

      // Reject pending requests
      const requestId = this.getLatestRequestId()
      if (requestId && this.pendingRequests.has(requestId)) {
        this.pendingRequests.get(requestId).reject(err)
        this.pendingRequests.delete(requestId)
      }
    }
  }

  /**
   * Handle a line of output from Claude (deprecated - for streaming mode)
   */
  handleOutput (line) {
    const parsed = parseClaudeLine(line)
    if (!parsed) return

    // Find which request this belongs to (use session_id or latest)
    const requestId =
      parsed.uuid || parsed.session_id || this.getLatestRequestId()

    if (!requestId || !this.pendingRequests.has(requestId)) {
      // Might be initialization output or orphaned - emit as event
      this.emit('output', parsed)
      return
    }

    const pending = this.pendingRequests.get(requestId)

    // Collect the event
    if (parsed.type === 'stream_event' || parsed.type === 'assistant') {
      pending.chunks.push(parsed)
      // Emit for streaming
      this.emit('chunk', { requestId, event: parsed })
    }

    // Check for completion
    if (parsed.type === 'result') {
      pending.resultEvent = parsed
      pending.resolve({
        chunks: pending.chunks,
        result: parsed
      })
      this.pendingRequests.delete(requestId)
    }
  }

  /**
   * Get the most recent request ID
   */
  getLatestRequestId () {
    const keys = Array.from(this.pendingRequests.keys())
    return keys.length > 0 ? keys[keys.length - 1] : null
  }

  /**
   * Send a message and get the response
   * Note: In non-streaming JSON mode, we spawn a new process for each request
   */
  async send (messages, model = 'claude-code', opts = {}) {
    this.lastUsed = Date.now()

    const { systemPrompt, prompt } = messagesToPrompt(messages)
    const fullSystem =
      systemPrompt + toolsToSystemPrompt(opts.tools, opts.toolChoice)
    const requestId = uuidv4()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        chunks: [],
        resultEvent: null
      })

      // Spawn new process with the prompt + requested model + system prompt
      this.spawn(prompt, model, fullSystem).catch(reject)
    })
  }

  /**
   * Send a message and stream the response.
   *
   * Spawns a dedicated `claude -p --output-format stream-json` process, parses
   * the NDJSON, and invokes onChunk() with OpenAI chat.completion.chunk objects:
   * a role chunk, content deltas, then a finish_reason chunk.
   *
   * When opts.tools is set, output is buffered (not streamed) so a tool-call
   * envelope can be detected and emitted as a tool_calls delta with
   * finish_reason 'tool_calls'; otherwise text deltas stream as they arrive.
   */
  async sendStreaming (messages, model, onChunk, opts = {}) {
    this.lastUsed = Date.now()

    const { systemPrompt, prompt } = messagesToPrompt(messages)
    const toolText = toolsToSystemPrompt(opts.tools, opts.toolChoice)
    const hasTools = toolText.length > 0
    const fullSystem = systemPrompt + toolText
    const completionId = `chatcmpl-${uuidv4().slice(0, 8)}`
    const args = this.buildClaudeArgs(prompt, model, {
      stream: true,
      systemPrompt: fullSystem
    })
    const { command, finalArgs } = this.resolveSpawnCommand(args)

    console.log(
      `[Worker ${this.conversationId}] ${USE_OAUTH ? 'OAuth' : 'API Key'} stream${
        hasTools ? '+tools' : ''
      }: ${command} -p "${prompt.substring(0, 30)}..."`
    )

    if (this.proc && this.proc.exitCode === null) {
      this.proc.kill()
    }

    const baseChunk = () => ({
      id: completionId,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{ index: 0, delta: {}, finish_reason: null }]
    })

    // Initial role chunk
    const roleChunk = baseChunk()
    roleChunk.choices[0].delta = { role: 'assistant' }
    onChunk(roleChunk)

    return new Promise((resolve, reject) => {
      const proc = spawn(command, finalArgs, {
        cwd: this.workspaceDir,
        env: { ...process.env, TERM: 'dumb', NO_COLOR: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
        shell: USE_OAUTH && process.platform === 'win32',
        windowsHide: true
      })
      this.proc = proc
      this.ready = true

      const rl = createInterface({ input: proc.stdout })
      let buffered = ''
      let sawResult = false
      let finished = false

      const emitContent = text => {
        const c = baseChunk()
        c.choices[0].delta = { content: text }
        onChunk(c)
      }
      const emitToolCalls = calls => {
        const c = baseChunk()
        c.choices[0].delta = {
          tool_calls: calls.map((tc, i) => ({
            index: i,
            id: tc.id,
            type: 'function',
            function: { name: tc.function.name, arguments: tc.function.arguments }
          }))
        }
        onChunk(c)
      }
      const finishWith = reason => {
        if (finished) return
        finished = true
        const c = baseChunk()
        c.choices[0].finish_reason = reason
        onChunk(c)
      }

      rl.on('line', line => {
        const parsed = parseClaudeLine(line)
        if (!parsed) return

        if (
          parsed.type === 'stream_event' &&
          parsed.event?.type === 'content_block_delta' &&
          parsed.event.delta?.type === 'text_delta' &&
          parsed.event.delta.text
        ) {
          buffered += parsed.event.delta.text
          if (!hasTools) emitContent(parsed.event.delta.text)
        } else if (parsed.type === 'result') {
          sawResult = true
          const text = buffered || parsed.result || ''
          if (hasTools) {
            const toolCalls = parseAssistantToolCalls(text)
            if (toolCalls) {
              emitToolCalls(toolCalls)
              finishWith('tool_calls')
            } else {
              if (text) emitContent(text)
              finishWith('stop')
            }
          } else {
            if (!buffered && parsed.result) emitContent(parsed.result)
            finishWith('stop')
          }
        }
      })

      proc.stderr.on('data', data => {
        const msg = data.toString().trim()
        if (msg) {
          console.error(`[Worker ${this.conversationId}] stderr: ${msg}`)
        }
      })

      proc.on('error', err => {
        this.proc = null
        this.ready = false
        reject(err)
      })

      proc.on('close', code => {
        this.proc = null
        this.ready = false
        rl.close()
        finishWith('stop') // safety net to always close the SSE stream
        if (code === 0 || sawResult) {
          resolve({ chunks: [], result: null })
        } else {
          reject(
            new Error(`Worker process exited with code ${code} before completing`)
          )
        }
      })
    })
  }

  /**
   * Kill the worker process
   */
  kill () {
    if (this.proc) {
      console.log(`[Worker ${this.conversationId}] Killing process`)
      this.proc.kill('SIGTERM')

      // Force kill after timeout
      setTimeout(() => {
        if (this.proc) {
          this.proc.kill('SIGKILL')
        }
      }, 5000)
    }
  }

  /**
   * Check if the worker is alive
   */
  isAlive () {
    return this.proc !== null && this.ready
  }

  /**
   * Get idle time in milliseconds
   */
  getIdleTime () {
    return Date.now() - this.lastUsed
  }
}
