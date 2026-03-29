import type { Response } from "express";
import pool, { genId } from "../db.js";

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface LogEntry {
  id: string;
  level: LogLevel;
  source: string;
  message: string;
  metadata: Record<string, any> | null;
  created_at: string;
}

interface SSEClient {
  res: Response;
  heartbeat: ReturnType<typeof setInterval>;
}

class LogService {
  private clients = new Map<string, SSEClient>();
  private cleanupTimer: ReturnType<typeof setInterval> | null = null;
  private dbAvailable = true;
  private consecutiveFailures = 0;
  private readonly MAX_FAILURES = 5;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly HEARTBEAT_MS = 15_000;
  private readonly CLEANUP_INTERVAL_MS = 10 * 60 * 1000;
  private readonly RETENTION_HOURS = 2;
  private readonly RECONNECT_MS = 30_000;

  constructor() {
    this.startCleanupScheduler();
  }

  private markDbDown(): void {
    if (this.dbAvailable) {
      this.dbAvailable = false;
      console.error("[LogService] DB unavailable — pausing DB writes. Will retry in 30s.");
    }
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.dbAvailable = true;
        this.consecutiveFailures = 0;
        console.log("[LogService] Re-enabling DB writes.");
      }, this.RECONNECT_MS);
    }
  }

  async log(
    level: LogLevel,
    source: string,
    message: string,
    metadata?: Record<string, any> | null,
  ): Promise<LogEntry> {
    const entry: LogEntry = {
      id: genId(),
      level,
      source,
      message,
      metadata: metadata || null,
      created_at: new Date().toISOString(),
    };

    if (this.dbAvailable) {
      pool
        .query(
          `INSERT INTO app_logs (id, level, source, message, metadata, created_at)
           VALUES ($1, $2, $3, $4, $5, $6)`,
          [
            entry.id,
            entry.level,
            entry.source,
            entry.message,
            entry.metadata ? JSON.stringify(entry.metadata) : null,
            entry.created_at,
          ],
        )
        .then(() => {
          this.consecutiveFailures = 0;
        })
        .catch(() => {
          this.consecutiveFailures++;
          if (this.consecutiveFailures >= this.MAX_FAILURES) {
            this.markDbDown();
          }
        });
    }

    this.broadcast(entry);

    return entry;
  }

  debug(source: string, message: string, metadata?: Record<string, any>) {
    return this.log("debug", source, message, metadata);
  }
  info(source: string, message: string, metadata?: Record<string, any>) {
    return this.log("info", source, message, metadata);
  }
  warn(source: string, message: string, metadata?: Record<string, any>) {
    return this.log("warn", source, message, metadata);
  }
  error(source: string, message: string, metadata?: Record<string, any>) {
    return this.log("error", source, message, metadata);
  }

  async query(opts: {
    level?: LogLevel;
    source?: string;
    search?: string;
    limit?: number;
    offset?: number;
    since?: string;
  }): Promise<{ logs: LogEntry[]; total: number }> {
    const conditions: string[] = [];
    const params: any[] = [];
    let idx = 1;

    if (opts.level) {
      conditions.push(`level = $${idx++}`);
      params.push(opts.level);
    }
    if (opts.source) {
      conditions.push(`source = $${idx++}`);
      params.push(opts.source);
    }
    if (opts.search) {
      conditions.push(`message ILIKE $${idx++}`);
      params.push(`%${opts.search}%`);
    }
    if (opts.since) {
      conditions.push(`created_at >= $${idx++}`);
      params.push(opts.since);
    }

    const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    const limit = Math.min(opts.limit || 100, 500);
    const offset = opts.offset || 0;

    const [logsResult, countResult] = await Promise.all([
      pool.query(
        `SELECT * FROM app_logs ${where} ORDER BY created_at DESC LIMIT ${limit} OFFSET ${offset}`,
        params,
      ),
      pool.query(`SELECT COUNT(*) FROM app_logs ${where}`, params),
    ]);

    return {
      logs: logsResult.rows,
      total: parseInt(countResult.rows[0].count, 10),
    };
  }

  registerClient(clientId: string, res: Response): void {
    this.removeClient(clientId);

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    if (typeof res.flushHeaders === "function") {
      res.flushHeaders();
    }

    const heartbeat = setInterval(() => {
      try {
        res.write(":heartbeat\n\n");
      } catch {
        this.removeClient(clientId);
      }
    }, this.HEARTBEAT_MS);

    this.clients.set(clientId, { res, heartbeat });

    res.on("close", () => {
      this.removeClient(clientId);
    });

    try {
      res.write(`event: connected\ndata: ${JSON.stringify({ clientId })}\n\n`);
    } catch {
      // ignore
    }
  }

  private removeClient(clientId: string): void {
    const client = this.clients.get(clientId);
    if (client) {
      clearInterval(client.heartbeat);
      this.clients.delete(clientId);
    }
  }

  private broadcast(entry: LogEntry): void {
    const data = JSON.stringify(entry);
    for (const [clientId, client] of this.clients) {
      try {
        client.res.write(`event: log\ndata: ${data}\n\n`);
      } catch {
        this.removeClient(clientId);
      }
    }
  }

  get connectedClients(): number {
    return this.clients.size;
  }

  async cleanup(): Promise<number> {
    const result = await pool.query(
      `DELETE FROM app_logs WHERE created_at < NOW() - INTERVAL '${this.RETENTION_HOURS} hours' RETURNING id`,
    );
    const count = result.rowCount || 0;
    if (count > 0) {
      console.log(`[LogService] Purged ${count} logs older than ${this.RETENTION_HOURS}h`);
    }
    return count;
  }

  private startCleanupScheduler(): void {
    setTimeout(() => this.cleanup().catch(() => {}), 10_000);

    this.cleanupTimer = setInterval(() => {
      this.cleanup().catch((err) => {
        console.error("[LogService] Cleanup failed:", err.message);
      });
    }, this.CLEANUP_INTERVAL_MS);
  }

  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    for (const [clientId] of this.clients) {
      this.removeClient(clientId);
    }
  }
}

export const logService = new LogService();
export default logService;
