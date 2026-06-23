#!/usr/bin/env node
/**
 * Onboarding helper for oauth mode (AUTH_MODE=oauth).
 *
 * Wraps `claude setup-token`, which runs the Claude subscription OAuth flow and
 * prints a long-lived authentication token. Run this once, then expose the
 * token to the server as the CLAUDE_CODE_OAUTH_TOKEN environment variable
 * (e.g. in your .env file, a Docker secret, or a Kubernetes Secret).
 *
 *   npm run login
 *
 * The flow:
 *   1. `claude setup-token` prints an authorization URL (and opens your browser
 *      if one is available).
 *   2. You approve access with your Claude account.
 *   3. The terminal prints a token like `sk-ant-oat01-...`.
 *   4. Set CLAUDE_CODE_OAUTH_TOKEN to that value for the API server.
 *
 * No API key and no ~/.claude credential mounting are required: the bundled
 * `claude` CLI reads CLAUDE_CODE_OAUTH_TOKEN directly.
 */
import { spawn } from 'child_process'

const isWindows = process.platform === 'win32'
const command = isWindows ? 'claude.cmd' : 'claude'

console.log('Starting Claude subscription login (claude setup-token)...\n')

const child = spawn(command, ['setup-token'], {
  stdio: 'inherit',
  shell: isWindows
})

child.on('error', err => {
  if (err.code === 'ENOENT') {
    console.error(
      '\nCould not find the `claude` CLI on PATH.\n' +
        'Install it first:  npm install -g @anthropic-ai/claude-code\n'
    )
  } else {
    console.error('\nFailed to run `claude setup-token`:', err.message)
  }
  process.exit(1)
})

child.on('exit', code => {
  if (code === 0) {
    console.log(
      '\nDone. Set the printed token as CLAUDE_CODE_OAUTH_TOKEN for the server:\n' +
        '  echo "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-..." >> .env\n' +
        'and make sure AUTH_MODE=oauth.\n'
    )
  }
  process.exit(code ?? 1)
})
