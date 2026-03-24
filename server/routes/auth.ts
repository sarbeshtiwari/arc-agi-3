import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import {
  authenticateToken,
  requireAdmin,
  hashPassword,
  verifyPassword,
  createToken,
} from "../middleware/auth.js";
import pool, { genId, queryOne } from "../db.js";

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
    res.json({
      id: user.id,
      username: user.username,
      email: user.email || null,
      is_admin: user.is_admin,
      is_active: user.is_active,
      allowed_pages: user.allowed_pages || [],
      created_at: user.created_at,
    });
  }),
);

// ──── POST /register (admin only) ────
router.post(
  "/register",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { username, password, email, is_admin, allowed_pages } = req.body;

    if (!username || typeof username !== "string" || username.length < 3 || username.length > 50) {
      res.status(400).json({ detail: "Username must be between 3 and 50 characters" });
      return;
    }
    if (!password || typeof password !== "string" || password.length < 6) {
      res.status(400).json({ detail: "Password must be at least 6 characters" });
      return;
    }

    const existing = await queryOne("SELECT id FROM users WHERE username = $1", [username]);

    if (existing) {
      res.status(400).json({ detail: "Username already exists" });
      return;
    }

    // Admins get all pages by default if none specified
    let pages: string[] = allowed_pages || [];
    if (is_admin && pages.length === 0) {
      pages = ["dashboard", "games", "upload", "requests", "users"];
    }

    const id = genId();
    const hashedPw = await hashPassword(password);

    await pool.query(
      `INSERT INTO users (id, username, hashed_password, email, is_admin, is_active, allowed_pages)
       VALUES ($1, $2, $3, $4, $5, TRUE, $6)`,
      [id, username, hashedPw, email || null, !!is_admin, JSON.stringify(pages)]
    );

    const newUser = await queryOne("SELECT * FROM users WHERE id = $1", [id]);

    res.json({
      id: newUser.id,
      username: newUser.username,
      email: newUser.email || null,
      is_admin: newUser.is_admin,
      is_active: newUser.is_active,
      allowed_pages: newUser.allowed_pages || [],
      created_at: newUser.created_at,
    });
  }),
);

export default router;
