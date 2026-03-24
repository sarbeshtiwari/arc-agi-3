import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireAdmin } from "../middleware/auth.js";
import pool, { queryOne, queryAll } from "../db.js";

const router = Router();

// ──── Public: Get a setting by key ────
router.get(
  "/:key",
  asyncHandler(async (req, res) => {
    const { key } = req.params;
    const row = await queryOne("SELECT key, value FROM app_settings WHERE key = $1", [key]);
    if (!row) return res.json({ key, value: null });
    res.json({ key: row.key, value: row.value });
  }),
);

// ──── Public: Get all settings ────
router.get(
  "/",
  asyncHandler(async (req, res) => {
    const rows = await queryAll("SELECT key, value FROM app_settings ORDER BY key");
    const settings: Record<string, string> = {};
    for (const row of rows) settings[row.key] = row.value;
    res.json(settings);
  }),
);

// ──── Admin: Update a setting ────
router.put(
  "/:key",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { key } = req.params;
    const { value } = req.body;
    if (value === undefined || value === null) {
      return res.status(400).json({ detail: "value is required" });
    }
    await pool.query(
      `INSERT INTO app_settings (key, value, updated_at) VALUES ($1, $2, NOW())
       ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()`,
      [key, String(value)],
    );
    res.json({ key, value: String(value) });
  }),
);

export default router;
