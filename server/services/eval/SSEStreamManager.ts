import type { Response } from "express";

interface SSEConnection {
  res: Response;
  heartbeat: ReturnType<typeof setInterval>;
  closed: boolean;
}

class SSEStreamManager {
  private connections = new Map<string, SSEConnection>();
  private readonly HEARTBEAT_INTERVAL_MS = 15_000;

  /**
   * Register a new SSE connection for the given session.
   * If a connection already exists for the session it is torn down first.
   */
  register(sessionId: string, res: Response): void {
    const existing = this.connections.get(sessionId);
    if (existing) {
      this.cleanup(sessionId);
    }

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    if (typeof res.flushHeaders === "function") {
      res.flushHeaders();
    }

    const heartbeat = setInterval(() => {
      if (this.connections.get(sessionId)?.closed) return;
      try {
        res.write(":heartbeat\n\n");
      } catch {
        // Connection already gone — cleanup will handle it
      }
    }, this.HEARTBEAT_INTERVAL_MS);

    const connection: SSEConnection = { res, heartbeat, closed: false };
    this.connections.set(sessionId, connection);

    res.on("close", () => {
      this.cleanup(sessionId);
    });
  }

  /**
   * Send a named SSE event with a JSON-serialised data payload.
   */
  sendEvent(sessionId: string, event: string, data: any): void {
    const conn = this.connections.get(sessionId);
    if (!conn || conn.closed) return;

    try {
      const serialized = JSON.stringify(data ?? {});
      conn.res.write(`event: ${event}\ndata: ${serialized}\n\n`);
    } catch {
      // Write may fail if the client disconnected between the check and the
      // write — that is fine, the 'close' handler will tidy up.
    }
  }

  /**
   * Gracefully close a session. Optionally sends a `stream.complete` event
   * with a summary payload before ending the response.
   */
  close(sessionId: string, summary?: any): void {
    const conn = this.connections.get(sessionId);
    if (!conn || conn.closed) return;

    if (summary !== undefined) {
      this.sendEvent(sessionId, "stream.complete", summary);
    } else {
      this.sendEvent(sessionId, "stream.complete", {});
    }

    this.cleanup(sessionId);
  }

  /**
   * Send a `stream.error` event and tear down the connection.
   */
  error(sessionId: string, code: string, message: string): void {
    const conn = this.connections.get(sessionId);
    if (!conn || conn.closed) return;

    this.sendEvent(sessionId, "stream.error", { code, message });
    this.cleanup(sessionId);
  }

  /**
   * Returns true when the session has an active (non-closed) connection.
   */
  isConnected(sessionId: string): boolean {
    const conn = this.connections.get(sessionId);
    return !!conn && !conn.closed;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private cleanup(sessionId: string): void {
    const conn = this.connections.get(sessionId);
    if (!conn) return;

    clearInterval(conn.heartbeat);

    if (!conn.closed) {
      conn.closed = true;
      try {
        conn.res.end();
      } catch {
        // Already ended — safe to ignore.
      }
    }

    this.connections.delete(sessionId);
  }
}

export const sseManager = new SSEStreamManager();
