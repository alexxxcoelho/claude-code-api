import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import { timingSafeEqual } from 'crypto'
import { v4 as uuidv4 } from 'uuid'
import { HybridPool } from './pool.js'

const app = express()
app.use(cors()) // Enable CORS for all routes
app.use(express.json({ limit: '10mb' }))

/**
 * Optional API-key authentication.
 *
 * If PROXY_API_KEY is set, every request (except /health) must present it as
 * either `Authorization: Bearer <key>` or `x-api-key: <key>`. This protects the
 * endpoint — which can spawn Claude Code workers and consume your subscription
 * quota — from untrusted callers when exposed beyond localhost.
 *
 * If PROXY_API_KEY is NOT set, the API stays open (backwards compatible) but a
 * loud warning is logged, because an unauthenticated, internet-reachable
 * instance lets anyone burn your quota.
 */
const PROXY_API_KEY = process.env.PROXY_API_KEY

if (!PROXY_API_KEY) {
  console.warn(
    '[Security] PROXY_API_KEY is not set — the API is UNAUTHENTICATED. ' +
      'Anyone who can reach this port can use your Claude subscription. ' +
      'Set PROXY_API_KEY to require a Bearer token.'
  )
}

function safeEqual (a, b) {
  const ba = Buffer.from(String(a))
  const bb = Buffer.from(String(b))
  if (ba.length !== bb.length) return false
  return timingSafeEqual(ba, bb)
}

app.use((req, res, next) => {
  if (req.path === '/health') return next()
  if (!PROXY_API_KEY) return next()

  const authHeader = req.headers['authorization'] || ''
  const presented = authHeader.startsWith('Bearer ')
    ? authHeader.slice(7)
    : req.headers['x-api-key'] || ''

  if (presented && safeEqual(presented, PROXY_API_KEY)) return next()

  return res.status(401).json({
    error: {
      message: 'Invalid or missing API key',
      type: 'authentication_error',
      code: 'invalid_api_key'
    }
  })
})

// Initialize the hybrid pool
const pool = new HybridPool()

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down...')
  pool.shutdown()
  process.exit(0)
})

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down...')
  pool.shutdown()
  process.exit(0)
})

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    ...pool.getStats()
  })
})

/**
 * List available models: the "claude-code" sentinel (CLI default model) plus
 * any client-facing ids declared in MODEL_ALIASES.
 */
let MODEL_ALIAS_IDS = []
try {
  if (process.env.MODEL_ALIASES) {
    MODEL_ALIAS_IDS = Object.keys(JSON.parse(process.env.MODEL_ALIASES))
  }
} catch (err) {
  console.error('[API] Invalid MODEL_ALIASES JSON, ignoring:', err.message)
}

app.get('/v1/models', (req, res) => {
  const created = Math.floor(Date.now() / 1000)
  const ids = ['claude-code', ...MODEL_ALIAS_IDS]
  res.json({
    object: 'list',
    data: ids.map(id => ({
      id,
      object: 'model',
      created,
      owned_by: 'anthropic',
      permission: [],
      root: id,
      parent: null
    }))
  })
})

/**
 * Get a specific model
 */
app.get('/v1/models/:model', (req, res) => {
  const modelId = req.params.model

  if (modelId === 'claude-code' || modelId.startsWith('claude-code:')) {
    res.json({
      id: modelId,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'anthropic'
    })
  } else {
    res.status(404).json({
      error: {
        message: `Model '${modelId}' not found`,
        type: 'invalid_request_error',
        code: 'model_not_found'
      }
    })
  }
})

/**
 * Extract conversation ID from request
 */
function getConversationId (req) {
  // UUID validation regex
  const uuidRegex =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

  // 1. Check X-Conversation-ID header
  if (req.headers['x-conversation-id']) {
    const providedId = req.headers['x-conversation-id']
    // If it's a valid UUID, use it
    if (uuidRegex.test(providedId)) {
      return providedId
    }
    // Otherwise, generate a UUID but include the provided ID in logs
    console.log(
      `[API] Invalid conversation ID provided: ${providedId}, generating new UUID`
    )
  }

  // 2. Check model name for suffix (claude-code:conversation-name)
  const model = req.body.model || ''
  if (model.includes(':')) {
    const parts = model.split(':')
    if (parts.length === 2 && parts[1]) {
      const modelId = parts[1]
      if (uuidRegex.test(modelId)) {
        return modelId
      }
    }
  }

  // 3. Generate new conversation ID
  return uuidv4()
}

/**
 * OpenAI-compatible chat completions endpoint
 */
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { messages, model = 'claude-code', stream = false } = req.body

    // Validate request
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'messages is required and must be a non-empty array',
          type: 'invalid_request_error',
          code: 'invalid_messages'
        }
      })
    }

    const conversationId = getConversationId(req)
    console.log(
      `[API] Chat completion request - conversation: ${conversationId}, stream: ${stream}`
    )

    if (stream) {
      // Streaming response
      res.setHeader('Content-Type', 'text/event-stream')
      res.setHeader('Cache-Control', 'no-cache')
      res.setHeader('Connection', 'keep-alive')
      res.setHeader('X-Conversation-ID', conversationId)

      try {
        for await (const chunk of pool.chatCompletionStream(
          conversationId,
          messages,
          model
        )) {
          res.write(chunk)
        }
        res.end()
      } catch (err) {
        console.error('[API] Streaming error:', err)
        // Try to send error in SSE format if connection still open
        if (!res.writableEnded) {
          res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`)
          res.end()
        }
      }
    } else {
      // Non-streaming response
      const response = await pool.chatCompletion(
        conversationId,
        messages,
        model
      )
      res.setHeader('X-Conversation-ID', conversationId)
      res.json(response)
    }
  } catch (err) {
    console.error('[API] Error:', err)
    res.status(500).json({
      error: {
        message: err.message || 'Internal server error',
        type: 'api_error',
        code: 'internal_error'
      }
    })
  }
})

/**
 * List active conversations (admin endpoint)
 */
app.get('/conversations', (req, res) => {
  res.json({
    conversations: pool.getActiveConversations(),
    stats: pool.getStats()
  })
})

/**
 * Evict a specific conversation (admin endpoint)
 */
app.delete('/conversations/:id', (req, res) => {
  const conversationId = req.params.id
  const evicted = pool.evictConversation(conversationId)

  if (evicted) {
    res.json({
      status: 'evicted',
      conversationId,
      note: 'Session saved to disk and can be resumed later'
    })
  } else {
    res.status(404).json({
      error: {
        message: `Conversation '${conversationId}' not found in active pool`,
        type: 'not_found_error'
      }
    })
  }
})

/**
 * Evict all conversations (admin endpoint)
 */
app.delete('/conversations', (req, res) => {
  const count = pool.workers.size
  pool.shutdown()

  // Restart the pool
  pool.startCleanup()

  res.json({
    status: 'all_evicted',
    count,
    note: 'All sessions saved to disk'
  })
})

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('[API] Unhandled error:', err)
  res.status(500).json({
    error: {
      message: 'Internal server error',
      type: 'api_error'
    }
  })
})

// Start server
const PORT = process.env.PORT || 3000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Claude Code API server listening on port ${PORT}`)
  console.log(`Health check: http://localhost:${PORT}/health`)
  console.log(`Chat endpoint: http://localhost:${PORT}/v1/chat/completions`)
})
