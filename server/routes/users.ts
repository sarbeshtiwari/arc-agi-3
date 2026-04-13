import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireAdmin, hashPassword, verifyPassword } from "../middleware/auth.js";
import pool, { genId, queryOne, queryAll, toJsonb } from "../db.js";
import { AppError } from "../middleware/errorHandler.js";
import { getUserLeadIds } from "../services/teamHelpers.js";

const router = Router();

const PROTECTED_USERNAME = process.env.PROTECTED_USERNAME || "";
const PROTECTED_SECRET_CODE = process.env.PROTECTED_SECRET_CODE || "";

function isProtected(user: any): boolean {
  return user.username === PROTECTED_USERNAME;
}

/** Format a user row from PostgreSQL for JSON response. */
async function formatUser(user: any) {
  const leadIds = await getUserLeadIds(user.id);
  return {
    id: user.id,
    username: user.username,
    display_name: user.display_name || null,
    email: user.email || null,
    is_admin: user.is_admin,
    is_active: user.is_active,
    role: user.role || 'tasker',
    team_lead_id: user.team_lead_id || null,
    team_lead_ids: leadIds,
    allowed_pages: user.allowed_pages ?? [],
    created_at: user.created_at,
  };
}

// ──── GET / — List all users (admin only) ────
router.get(
  "/",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const users = await queryAll(
      "SELECT * FROM users ORDER BY created_at DESC",
    );

    const result = await Promise.all(users.map(formatUser));
    res.json(result);
  }),
);

// ──── GET /:userId — Get single user (admin only) ────
router.get(
  "/:userId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { userId } = req.params;
    const user = await queryOne("SELECT * FROM users WHERE id = $1", [userId]);

    if (!user) {
      throw new AppError("User not found", 404);
    }

    res.json(await formatUser(user));
  }),
);

// ──── PUT /:userId — Update user (admin only, protected user blocked) ────
router.put(
  "/:userId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { userId } = req.params;
    const user = await queryOne("SELECT * FROM users WHERE id = $1", [userId]);

    if (!user) {
      throw new AppError("User not found", 404);
    }

    if (isProtected(user)) {
      throw new AppError("This user is protected and cannot be modified", 403);
    }

    const { email, display_name, is_admin, is_active, allowed_pages, password, role, team_lead_ids } = req.body;

    const updates: string[] = [];
    const values: any[] = [];
    let paramIndex = 1;

    if (email !== undefined && email !== null) {
      updates.push(`email = $${paramIndex++}`);
      values.push(email);
    }
    if (display_name !== undefined) {
      updates.push(`display_name = $${paramIndex++}`);
      values.push(display_name);
    }
    if (is_admin !== undefined && is_admin !== null) {
      updates.push(`is_admin = $${paramIndex++}`);
      values.push(is_admin);
    }
    if (is_active !== undefined && is_active !== null) {
      updates.push(`is_active = $${paramIndex++}`);
      values.push(is_active);
    }
    if (allowed_pages !== undefined && allowed_pages !== null) {
      updates.push(`allowed_pages = $${paramIndex++}`);
      values.push(toJsonb(allowed_pages));
    }
    if (password !== undefined && password !== null) {
      const hashed = await hashPassword(password);
      updates.push(`hashed_password = $${paramIndex++}`);
      values.push(hashed);
    }
    if (role !== undefined && role !== null) {
      updates.push(`role = $${paramIndex++}`);
      values.push(role);
    }

    if (Array.isArray(team_lead_ids)) {
      await pool.query(`DELETE FROM user_team_leads WHERE user_id = $1`, [userId]);
      for (const leadId of team_lead_ids) {
        await pool.query(
          `INSERT INTO user_team_leads (id, user_id, lead_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
          [genId(), userId, leadId]
        );
      }
      const primaryLead = team_lead_ids[0] || null;
      updates.push(`team_lead_id = $${paramIndex++}`);
      values.push(primaryLead);
    }

    if (updates.length > 0) {
      updates.push("updated_at = NOW()");
      values.push(userId);
      await pool.query(
        `UPDATE users SET ${updates.join(", ")} WHERE id = $${paramIndex}`,
        values,
      );
    }

    const updated = await queryOne("SELECT * FROM users WHERE id = $1", [userId]);
    res.json(await formatUser(updated));
  }),
);

// ──── DELETE /:userId — Delete user (admin only, protected/self blocked) ────
router.delete(
  "/:userId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { userId } = req.params;
    const user = await queryOne("SELECT * FROM users WHERE id = $1", [userId]);

    if (!user) {
      throw new AppError("User not found", 404);
    }

    if (isProtected(user)) {
      throw new AppError("This user is protected and cannot be deleted", 403);
    }

    if (user.id === req.user.id) {
      throw new AppError("Cannot delete yourself", 400);
    }

    await pool.query("DELETE FROM users WHERE id = $1", [userId]);

    res.json({ detail: "User deleted" });
  }),
);

// ──── POST /protected/change-password — Protected admin password change with secret code ────
router.post(
  "/protected/change-password",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const currentUser = req.user;

    if (currentUser.username !== PROTECTED_USERNAME) {
      throw new AppError("Only the protected user can change their own password", 403);
    }

    const { secret_code, new_password } = req.body;

    if (secret_code !== PROTECTED_SECRET_CODE) {
      throw new AppError("Invalid secret code", 403);
    }

    if (!new_password || new_password.length < 6) {
      throw new AppError("Password must be at least 6 characters", 400);
    }

    const hashed = await hashPassword(new_password);
    await pool.query(
      "UPDATE users SET hashed_password = $1, updated_at = NOW() WHERE id = $2",
      [hashed, currentUser.id],
    );

    const updated = await queryOne("SELECT * FROM users WHERE id = $1", [currentUser.id]);
    res.json(await formatUser(updated));
  }),
);

router.put(
  "/me/change-password",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { current_password, new_password } = req.body;

    if (!current_password || !new_password) {
      throw new AppError("current_password and new_password are required", 400);
    }

    if (new_password.length < 6) {
      throw new AppError("Password must be at least 6 characters", 400);
    }

    const user = await queryOne("SELECT * FROM users WHERE id = $1", [req.user.id]);
    if (!user) throw new AppError("User not found", 404);

    const valid = await verifyPassword(current_password, user.hashed_password);
    if (!valid) {
      throw new AppError("Current password is incorrect", 403);
    }

    const hashed = await hashPassword(new_password);
    await pool.query(
      "UPDATE users SET hashed_password = $1, updated_at = NOW() WHERE id = $2",
      [hashed, req.user.id],
    );

    res.json({ ok: true });
  }),
);

export default router;
