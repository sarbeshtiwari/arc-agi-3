# Multi-stage build: Node.js + Python for ARC-AGI-3 Platform
# ──────────────────────────────────────────────────────────

# Stage 1: Build the frontend + bundle the server
FROM node:20-slim AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
# Skip postinstall during build (Python not available yet)
RUN npm ci --ignore-scripts

COPY . .
RUN npm run build


# Stage 2: Production image with Node.js + Python 3.13 (arc-agi requires >=3.12)
FROM nikolaik/python-nodejs:python3.13-nodejs20-slim

WORKDIR /app

# Copy built artifacts and source
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/server/python ./server/python
COPY --from=builder /app/server/requirements.txt ./server/requirements.txt
COPY --from=builder /app/environment_files ./environment_files

# Install Python dependencies
RUN pip install -r server/requirements.txt

# Write .python-bin marker so GamePythonBridge finds Python
RUN which python3 > .python-bin

# Expose port (Render uses PORT env var)
EXPOSE 10000

ENV NODE_ENV=production
ENV PORT=10000

CMD ["node", "dist/index.js"]
