import { ClaudeWorker } from './worker.js'
import { v4 as uuidv4 } from 'uuid'
import { claudeResultToOpenai, formatSSE, formatSSEDone } from './translator.js'

/**
 * Hybrid pool manager for Claude Code workers
 * - Keeps active conversations in memory (hot workers)
 * - Evicts least recently used when at capacity
 * - Sessions persist to disk for later resume
 */
export class HybridPool {
  constructor (options = {}) {
    this.maxWorkers = options.poolSize || parseInt(process.env.POOL_SIZE) || 5
    this.idleTimeout =
      options.idleTimeoutMs || parseInt(process.env.IDLE_TIMEOUT_MS) || 300000
    this.workspaceDir =
      options.workspaceDir || process.env.WORKSPACE_DIR || '/workspace'
    this.workers = new Map() // conversationId -> ClaudeWorker
    this.cleanupInterval = null

    // Start periodic cleanup
    this.startCleanup()

    console.log(
      `[Pool] Initialized with maxWorkers=${this.maxWorkers}, idleTimeout=${this.idleTimeout}ms`
    )
  }

  /**
   * Start the idle worker cleanup interval
   */
  startCleanup () {
    if (this.cleanupInterval) return

    this.cleanupInterval = setInterval(() => {
      this.cleanupIdleWorkers()
    }, 60000) // Check every minute
  }

  /**
   * Stop the cleanup interval
   */
  stopCleanup () {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval)
      this.cleanupInterval = null
    }
  }

  /**
   * Clean up workers that have been idle too long
   */
  cleanupIdleWorkers () {
    const now = Date.now()
    console.log(`[Pool] Cleanup check - current workers: ${this.workers.size}`)

    for (const [convId, worker] of this.workers) {
      const idleTime = worker.getIdleTime()
      console.log(
        `[Pool] Worker ${convId.slice(0, 8)}: idle=${Math.floor(
          idleTime / 1000
        )}s, threshold=${Math.floor(this.idleTimeout / 1000)}s, lastUsed=${
          worker.lastUsed
        }, now=${now}`
      )

      if (idleTime > this.idleTimeout) {
        console.log(
          `[Pool] Evicting idle worker: ${convId} (idle for ${Math.floor(
            idleTime / 1000
          )}s)`
        )
        worker.kill()
        this.workers.delete(convId)
      }
    }
  }

  /**
   * Get or create a worker for a conversation
   */
  async getOrCreateWorker (conversationId) {
    // Check for existing hot worker
    if (this.workers.has(conversationId)) {
      const worker = this.workers.get(conversationId)
      if (worker.isAlive()) {
        console.log(`[Pool] Reusing hot worker for: ${conversationId}`)
        return worker
      } else {
        // Dead worker, remove it
        this.workers.delete(conversationId)
      }
    }

    // Check capacity
    if (this.workers.size >= this.maxWorkers) {
      this.evictLRU()
    }

    // Create new worker
    console.log(`[Pool] Creating new worker for: ${conversationId}`)
    const worker = new ClaudeWorker(conversationId, {
      workspaceDir: this.workspaceDir
    })

    // Handle worker close
    worker.on('close', () => {
      this.workers.delete(conversationId)
    })

    // Note: We don't spawn here anymore - spawn happens in worker.send() with the prompt

    this.workers.set(conversationId, worker)
    return worker
  }

  /**
   * Evict the least recently used worker
   */
  evictLRU () {
    let oldest = null
    let oldestTime = Infinity

    for (const [convId, worker] of this.workers) {
      if (worker.lastUsed < oldestTime) {
        oldestTime = worker.lastUsed
        oldest = convId
      }
    }

    if (oldest) {
      console.log(`[Pool] Evicting LRU worker: ${oldest}`)
      const worker = this.workers.get(oldest)
      worker.kill()
      this.workers.delete(oldest)
    }
  }

  /**
   * Handle a chat completion request (non-streaming)
   */
  async chatCompletion (conversationId, messages, model, opts = {}) {
    const worker = await this.getOrCreateWorker(conversationId)
    const completionId = `chatcmpl-${uuidv4().slice(0, 8)}`

    try {
      const { chunks, result } = await worker.send(messages, model, opts)
      return claudeResultToOpenai(result, chunks, completionId, model, {
        tools: opts.tools
      })
    } catch (err) {
      console.error(`[Pool] Error in chat completion:`, err)
      throw err
    }
  }

  /**
   * Handle a streaming chat completion request
   * Returns an async generator of SSE strings
   */
  async *chatCompletionStream (conversationId, messages, model, opts = {}) {
    const worker = await this.getOrCreateWorker(conversationId)
    const completionId = `chatcmpl-${uuidv4().slice(0, 8)}`

    // Use a queue to collect chunks
    const chunks = []
    let done = false
    let error = null

    // Start the request
    const requestPromise = worker
      .sendStreaming(
        messages,
        model,
        chunk => {
          chunks.push(chunk)
        },
        opts
      )
      .then(() => {
        done = true
      })
      .catch(err => {
        error = err
        done = true
      })

    // Yield chunks as they arrive
    let yielded = 0
    while (!done || yielded < chunks.length) {
      if (yielded < chunks.length) {
        yield formatSSE(chunks[yielded])
        yielded++
      } else {
        // Wait a bit for more chunks
        await new Promise(resolve => setTimeout(resolve, 10))
      }
    }

    if (error) {
      throw error
    }

    // Send done signal
    yield formatSSEDone()
  }

  /**
   * Get list of active conversations
   */
  getActiveConversations () {
    const active = []
    for (const [convId, worker] of this.workers) {
      active.push({
        id: convId,
        lastUsed: worker.lastUsed,
        idleSeconds: Math.floor(worker.getIdleTime() / 1000),
        alive: worker.isAlive()
      })
    }
    return active
  }

  /**
   * Force evict a specific conversation
   */
  evictConversation (conversationId) {
    const worker = this.workers.get(conversationId)
    if (worker) {
      worker.kill()
      this.workers.delete(conversationId)
      return true
    }
    return false
  }

  /**
   * Shutdown all workers
   */
  shutdown () {
    console.log('[Pool] Shutting down all workers')
    this.stopCleanup()

    for (const [convId, worker] of this.workers) {
      worker.kill()
    }
    this.workers.clear()
  }

  /**
   * Get pool statistics
   */
  getStats () {
    return {
      activeWorkers: this.workers.size,
      maxWorkers: this.maxWorkers,
      idleTimeoutMs: this.idleTimeout,
      workspaceDir: this.workspaceDir
    }
  }
}
