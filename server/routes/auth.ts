import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import {
  authenticateToken,
  requireAdmin,
  hashPassword,
  verifyPassword,
  createToken,
  requireRole,
} from "../middleware/auth.js";
import pool, { genId, queryOne } from "../db.js";
import { getUserLeadIds } from "../services/teamHelpers.js";

const router = Router();

// ──── POST /login ────
router.post(
  "/login",
  asyncHandler(async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
      res.status(400).json({ detail: "Username and password are required" });
      return;
    }

    const user = await queryOne("SELECT * FROM users WHERE username = $1", [username]);

    if (!user || !(await verifyPassword(password, user.hashed_password))) {
      res.status(401).json({ detail: "Invalid username or password" });
      return;
    }

    if (!user.is_active) {
      res.status(403).json({ detail: "Account is deactivated" });
      return;
    }

    const token = createToken(user.id);
    res.json({ access_token: token, token_type: "bearer" });
  }),
);

// ──── GET /me ────
router.get(
  "/me",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const user = req.user;
    const teamLeadIds = await getUserLeadIds(user.id);
    res.json({
      id: user.id,
      username: user.username,
      display_name: user.display_name || null,
      email: user.email || null,
      is_admin: user.is_admin,
      is_active: user.is_active,
      role: user.role || 'tasker',
      team_lead_id: user.team_lead_id || null,
      team_lead_ids: teamLeadIds,
      allowed_pages: user.allowed_pages || [],
      created_at: user.created_at,
    });
  }),
);

// ──── POST /register (admin only) ────
router.post(
  "/register",
  authenticateToken,
  requireRole('super_admin', 'pl', 'ql'),
  asyncHandler(async (req, res) => {
    const { username, password, display_name, email, is_admin, role, team_lead_ids, allowed_pages } = req.body;

    if (!username || typeof username !== "string" || username.length < 3 || username.length > 50) {
      res.status(400).json({ detail: "Username must be between 3 and 50 characters" });
      return;
    }
    if (!password || typeof password !== "string" || password.length < 6) {
      res.status(400).json({ detail: "Password must be at least 6 characters" });
      return;
    }

    const callerRole = req.user.role || 'tasker';
    const targetRole = role || 'tasker';

    if (callerRole === 'ql' && targetRole !== 'tasker') {
      res.status(403).json({ detail: "QLs can only create Tasker accounts" });
      return;
    }

    if (callerRole === 'pl' && targetRole !== 'tasker' && targetRole !== 'ql') {
      res.status(403).json({ detail: "PLs can only create Tasker and QL accounts" });
      return;
    }

    const existing = await queryOne("SELECT id FROM users WHERE username = $1", [username]);

    if (existing) {
      res.status(400).json({ detail: "Username already exists" });
      return;
    }

    const isSuperAdmin = targetRole === 'super_admin';
    let pages: string[] = allowed_pages || [];
    if (isSuperAdmin && pages.length === 0) {
      pages = ["dashboard", "games", "upload", "requests", "users"];
    }

    const id = genId();
    const hashedPw = await hashPassword(password);

    const firstLeadId = callerRole === 'ql' ? req.user.id : (Array.isArray(team_lead_ids) && team_lead_ids[0]) || null;

    await pool.query(
      `INSERT INTO users (id, username, hashed_password, display_name, email, is_admin, is_active, role, team_lead_id, allowed_pages)
       VALUES ($1, $2, $3, $4, $5, $6, TRUE, $7, $8, $9)`,
      [id, username, hashedPw, display_name || null, email || null, !!is_admin || isSuperAdmin, targetRole, firstLeadId, JSON.stringify(pages)]
    );

    let leadIds: string[] = [];
    if (callerRole === 'ql') {
      leadIds = [req.user.id];
    } else if (Array.isArray(team_lead_ids) && team_lead_ids.length > 0) {
      leadIds = team_lead_ids;
    }

    for (const leadId of leadIds) {
      await pool.query(
        `INSERT INTO user_team_leads (id, user_id, lead_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
        [genId(), id, leadId]
      );
    }

    const newUser = await queryOne("SELECT * FROM users WHERE id = $1", [id]);
    const newLeadIds = await getUserLeadIds(id);

    res.json({
      id: newUser.id,
      username: newUser.username,
      display_name: newUser.display_name || null,
      email: newUser.email || null,
      is_admin: newUser.is_admin,
      is_active: newUser.is_active,
      role: newUser.role || 'tasker',
      team_lead_id: newUser.team_lead_id || null,
      team_lead_ids: newLeadIds,
      allowed_pages: newUser.allowed_pages || [],
      created_at: newUser.created_at,
    });
  }),
);

export default router;
