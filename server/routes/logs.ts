import { Router } from "express";
import { authenticateToken, requireAdmin } from "../middleware/auth.js";
import { logService } from "../services/LogService.js";
import type { LogLevel } from "../services/LogService.js";

const router = Router();

router.get(
  "/",
  authenticateToken,
  requireAdmin,
  async (req, res, next) => {
    try {
      const { level, source, search, limit, offset, since } = req.query;

      const result = await logService.query({
        level: level as LogLevel | undefined,
        source: source as string | undefined,
        search: search as string | undefined,
        limit: limit ? parseInt(limit as string, 10) : 100,
        offset: offset ? parseInt(offset as string, 10) : 0,
        since: since as string | undefined,
      });

      res.json(result);
    } catch (err) {
      next(err);
    }
  },
);

router.get(
  "/stream",
  authenticateToken,
  requireAdmin,
  (req, res) => {
    const clientId = `log_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    logService.registerClient(clientId, res);
  },
);

router.get(
  "/sources",
  authenticateToken,
  requireAdmin,
  async (_req, res, next) => {
    try {
      const { rows } = await (await import("../db.js")).default.query(
        "SELECT DISTINCT source FROM app_logs ORDER BY source",
      );
      res.json(rows.map((r: any) => r.source));
    } catch (err) {
      next(err);
    }
  },
);

router.delete(
  "/cleanup",
  authenticateToken,
  requireAdmin,
  async (_req, res, next) => {
    try {
      const deleted = await logService.cleanup();
      res.json({ deleted });
    } catch (err) {
      next(err);
    }
  },
);

router.get(
  "/stats",
  authenticateToken,
  requireAdmin,
  async (_req, res, next) => {
    try {
      const pool = (await import("../db.js")).default;
      const [total, byLevel, connectedClients] = await Promise.all([
        pool.query("SELECT COUNT(*) FROM app_logs"),
        pool.query(
          "SELECT level, COUNT(*) as count FROM app_logs GROUP BY level ORDER BY level",
        ),
        Promise.resolve(logService.connectedClients),
      ]);

      res.json({
        total: parseInt(total.rows[0].count, 10),
        by_level: Object.fromEntries(
          byLevel.rows.map((r: any) => [r.level, parseInt(r.count, 10)]),
        ),
        connected_clients: connectedClients,
      });
    } catch (err) {
      next(err);
    }
  },
);

export default router;
