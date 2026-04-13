import type { Express } from "express";
import { createServer, type Server } from "http";
import os from "os";
import { errorHandler } from "./middleware/errorHandler.js";
import pool, { initDB, queryOne, queryAll } from "./db.js";

// Import route modules
import authRouter from "./routes/auth.js";
import gamesRouter from "./routes/games.js";
import playerRouter, { getActiveBridgeStats, killAllBridges } from "./routes/player.js";
import analyticsRouter from "./routes/analytics.js";
import usersRouter from "./routes/users.js";
import requestsRouter from "./routes/requests.js";
import evalRouter from "./routes/eval.js";
import settingsRouter from "./routes/settings.js";
import logsRouter from "./routes/logs.js";
import approvalRouter from "./routes/approval.js";
import teamsRouter from "./routes/teams.js";
import notificationsRouter from "./routes/notifications.js";

// Import startup helpers
import { ensureDefaultAdmin, authenticateToken, requireAdmin } from "./middleware/auth.js";

const SERVER_START_TIME = Date.now();

function formatUptime(ms: number): string {
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${d}d ${h}h ${m}m`;
}

export async function registerRoutes(app: Express): Promise<Server> {
  // ──── Startup: init DB tables ────
  await initDB();

  // ──── Startup: ensure default admin exists ────
  await ensureDefaultAdmin();

  // ──── Basic health check (public) ────
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", app: "ARC-AGI Internal", timestamp: new Date().toISOString() });
  });

  // ──── Detailed system status (admin only) ────
  app.get("/api/admin/system/status", authenticateToken, requireAdmin, async (req, res) => {
    try {
      const mem = process.memoryUsage();
      const uptimeMs = Date.now() - SERVER_START_TIME;

      const totalConns = Number((pool as any).totalCount) || 0;
      const idleConns = Number((pool as any).idleCount) || 0;
      const waitingConns = Number((pool as any).waitingCount) || 0;
      const dbPool = {
        total: totalConns,
        active: Math.max(0, totalConns - idleConns),
        idle: idleConns,
        waiting: waitingConns,
        max: 50,
      };

      const [games, sessions, tempSessions, users, logs, requests, analytics] = await Promise.all([
        queryOne("SELECT COUNT(*)::int AS count FROM games"),
        queryOne("SELECT COUNT(*)::int AS count FROM play_sessions"),
        queryOne("SELECT COUNT(*)::int AS count FROM temp_game_sessions"),
        queryOne("SELECT COUNT(*)::int AS count FROM users"),
        queryOne("SELECT COUNT(*)::int AS count FROM app_logs"),
        queryOne("SELECT COUNT(*)::int AS count FROM game_requests"),
        queryOne("SELECT COUNT(*)::int AS count FROM game_analytics"),
      ]);

      const expired = await queryOne(
        "SELECT COUNT(*)::int AS count FROM temp_game_sessions WHERE expires_at < NOW()"
      );
      const activeGames = await queryOne("SELECT COUNT(*)::int AS count FROM games WHERE is_active = true");
      const dbSize = await queryOne(
        "SELECT pg_size_pretty(pg_database_size(current_database())) AS size"
      );
      const oldestLog = await queryOne("SELECT created_at FROM app_logs ORDER BY created_at ASC LIMIT 1");

      res.json({
        status: "ok",
        timestamp: new Date().toISOString(),
        server: {
          uptime_ms: uptimeMs,
          uptime_human: formatUptime(uptimeMs),
          node_version: process.version,
          platform: os.platform(),
          arch: os.arch(),
          hostname: os.hostname(),
        },
        memory: {
          rss_mb: Math.round(mem.rss / 1024 / 1024),
          heap_used_mb: Math.round(mem.heapUsed / 1024 / 1024),
          heap_total_mb: Math.round(mem.heapTotal / 1024 / 1024),
          external_mb: Math.round(mem.external / 1024 / 1024),
          system_total_mb: Math.round(os.totalmem() / 1024 / 1024),
          system_free_mb: Math.round(os.freemem() / 1024 / 1024),
        },
        database: {
          pool: dbPool,
          size: dbSize?.size ?? "unknown",
          tables: {
            games: games?.count ?? 0,
            active_games: activeGames?.count ?? 0,
            play_sessions: sessions?.count ?? 0,
            temp_sessions: tempSessions?.count ?? 0,
            expired_temp_sessions: expired?.count ?? 0,
            users: users?.count ?? 0,
            logs: logs?.count ?? 0,
            requests: requests?.count ?? 0,
            analytics: analytics?.count ?? 0,
          },
          oldest_log: oldestLog?.created_at ?? null,
        },
        cleanup: {
          temp_session_ttl: "24 hours",
          auto_cleanup_interval: "1 hour",
          expired_count: expired?.count ?? 0,
        },
        bridges: getActiveBridgeStats(),
      });
    } catch (err: any) {
      res.status(500).json({ detail: err.message });
    }
  });

  // ──── Purge expired temp sessions (admin only) ────
  app.post("/api/admin/system/cleanup-expired", authenticateToken, requireAdmin, async (req, res) => {
    try {
      const result = await pool.query("DELETE FROM temp_game_sessions WHERE expires_at < NOW()");
      res.json({ deleted: result.rowCount ?? 0 });
    } catch (err: any) {
      res.status(500).json({ detail: err.message });
    }
  });

  app.post("/api/admin/system/kill-bridges", authenticateToken, requireAdmin, (req, res) => {
    try {
      const result = killAllBridges();
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ detail: err.message });
    }
  });

  // ──── API routes ────
  app.use("/api/auth", authRouter);
  app.use("/api/games", gamesRouter);
  app.use("/api/player", playerRouter);
  app.use("/api/analytics", analyticsRouter);
  app.use("/api/users", usersRouter);
  app.use("/api/requests", requestsRouter);
  app.use("/api/eval", evalRouter);
  app.use("/api/settings", settingsRouter);
  app.use("/api/admin/logs", logsRouter);
  app.use("/api/approval", approvalRouter);
  app.use("/api/teams", teamsRouter);
  app.use("/api/notifications", notificationsRouter);

  // ──── Error handler (must be last) ────
  app.use(errorHandler);

  return createServer(app);
}
