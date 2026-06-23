FROM node:20-bookworm

# Install system dependencies (based on claude-code devcontainer)
RUN apt-get update && apt-get install -y \
    git \
    zsh \
    vim \
    jq \
    fzf \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Create app directory
WORKDIR /app

# Copy package files first for better caching
COPY package.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Create workspace + Claude config dirs owned by the non-root `node` user
# (uid 1000, shipped with the official node image). In oauth mode the
# subscription credentials are read from $HOME/.claude.
RUN mkdir -p /workspace /home/node/.claude \
    && chown -R node:node /app /workspace /home/node/.claude

# Environment variables with defaults
ENV NODE_ENV=production
ENV HOME=/home/node
ENV PORT=3000
ENV POOL_SIZE=5
ENV IDLE_TIMEOUT_MS=300000
ENV WORKSPACE_DIR=/workspace

# Drop privileges: never run the worker pool as root
USER node

# Expose API port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start the server
CMD ["node", "server.js"]
