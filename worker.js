import { spawn } from 'child_process'
import { createInterface } from 'readline'
import { EventEmitter } from 'events'
import { v4 as uuidv4 } from 'uuid'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import {
  openaiToClaudeInput,
  claudeToOpenaiChunk,
  claudeResultToOpenai,
  parseClaudeLine
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
   * Spawn the Claude Code CLI process
   * Note: We spawn a new process for each request in non-streaming mode
   */
  async spawn (prompt, model) {
    if (this.proc && this.proc.exitCode === null) {
      // Kill existing process if still running
      this.proc.kill()
    }

    // Note: We use --output-format json (not stream-json) for simpler parsing
    //
    // Security: tool access and permission bypass are OPT-IN, not the default.
    //   CLAUDE_TOOLS  -> value passed to `--tools` ("" disables ALL tools, the
    //                    safe default; "default" enables all; or a list like
    //                    "Read,Edit"). With tools disabled the worker can only
    //                    generate text, removing the RCE surface for an exposed
    //                    HTTP endpoint.
    //   CLAUDE_SKIP_PERMISSIONS=true -> restores the old
    //                    `--dangerously-skip-permissions` behaviour. Only
    //                    meaningful when tools are enabled. Never enable this on
    //                    an endpoint reachable by untrusted clients.
    //
    // NOTE: `--tools` is variadic, so it must be followed by another flag and
    // never sit immediately before the prompt, or it swallows the prompt as a
    // tool name. The prompt is always passed LAST as the lone positional arg.
    const toolsSetting = process.env.CLAUDE_TOOLS ?? ''
    const skipPermissions = process.env.CLAUDE_SKIP_PERMISSIONS === 'true'

    const args = [
      '-p',
      '--tools',
      toolsSetting,
      '--output-format',
      'json',
      '--session-id',
      this.conversationId
    ]
    const resolvedModel = resolveModel(model)
    if (resolvedModel) {
      args.push('--model', resolvedModel)
    }
    if (skipPermissions) {
      args.push('--dangerously-skip-permissions')
    }
    args.push(prompt) // prompt MUST be the final argument

    // Use OAuth mode (global claude) or API key mode (local installation)
    const command = USE_OAUTH
      ? process.platform === 'win32'
        ? 'claude.cmd'
        : 'claude'
      : 'node'
    const finalArgs = USE_OAUTH ? args : [CLAUDE_CLI_PATH, ...args]

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
  async send (messages, model = 'claude-code') {
    this.lastUsed = Date.now()

    // Build prompt from messages
    let prompt = ''
    for (const msg of messages) {
      if (msg.role === 'system') {
        prompt += `System: ${msg.content}\n\n`
      } else if (msg.role === 'user') {
        prompt += msg.content
      } else if (msg.role === 'assistant') {
        // Skip assistant messages - Claude maintains history via session
      }
    }

    const requestId = uuidv4()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        chunks: [],
        resultEvent: null
      })

      // Spawn new process with the prompt + requested model
      this.spawn(prompt, model).catch(reject)
    })
  }

  /**
   * Send a message with streaming callback
   */
  async sendStreaming (messages, model, onChunk) {
    if (!this.proc || !this.ready) {
      await this.spawn()
    }

    this.lastUsed = Date.now()

    const claudeInput = openaiToClaudeInput(messages, this.conversationId)
    const requestId = claudeInput.uuid
    const completionId = `chatcmpl-${uuidv4().slice(0, 8)}`

    // Send initial role chunk
    onChunk(claudeToOpenaiChunk({ type: 'message_start' }, completionId, model))

    return new Promise((resolve, reject) => {
      const pending = {
        resolve,
        reject,
        chunks: [],
        resultEvent: null
      }

      this.pendingRequests.set(requestId, pending)

      // Listen for chunks specific to this request
      const chunkHandler = ({ requestId: chunkReqId, event }) => {
        if (chunkReqId === requestId) {
          const openaiChunk = claudeToOpenaiChunk(event, completionId, model)
          if (
            openaiChunk.choices[0].delta.content ||
            openaiChunk.choices[0].finish_reason
          ) {
            onChunk(openaiChunk)
          }
        }
      }

      this.on('chunk', chunkHandler)

      // Modify resolve to clean up listener
      const originalResolve = pending.resolve
      pending.resolve = result => {
        this.off('chunk', chunkHandler)

        // Send final chunk with finish_reason
        onChunk(claudeToOpenaiChunk({ type: 'result' }, completionId, model))

        originalResolve(result)
      }

      pending.reject = err => {
        this.off('chunk', chunkHandler)
        reject(err)
      }

      // Send the message
      const inputLine = JSON.stringify(claudeInput) + '\n'
      this.proc.stdin.write(inputLine, err => {
        if (err) {
          this.off('chunk', chunkHandler)
          this.pendingRequests.delete(requestId)
          reject(err)
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
