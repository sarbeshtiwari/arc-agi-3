import type { Request, Response, NextFunction } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import pool, { genId, queryOne } from "../db.js";

// ──── Augment Express Request with user ────
declare global {
  namespace Express {
    interface Request {
      user?: any;
    }
  }
}

const SECRET_KEY = process.env.SECRET_KEY || "change-me";
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || "1440m"; // 24h default, matches Python ACCESS_TOKEN_EXPIRE_MINUTES=1440
const ALGORITHM = "HS256";

// ──── Password helpers ────

export async function hashPassword(password: string): Promise<string> {
  const salt = await bcrypt.genSalt(10);
  return bcrypt.hash(password, salt);
}

export async function verifyPassword(
  plainPassword: string,
  hashedPassword: string,
): Promise<boolean> {
  return bcrypt.compare(plainPassword, hashedPassword);
}

// ──── Token helper ────

export function createToken(userId: string): string {
  return jwt.sign({ sub: userId }, SECRET_KEY, {
    algorithm: ALGORITHM,
    expiresIn: JWT_EXPIRES_IN,
  });
}

// ──── Middleware: authenticate JWT ────

export function authenticateToken(
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const authHeader = req.headers.authorization;
  const token = authHeader?.startsWith("Bearer ")
    ? authHeader.slice(7)
    : (req.query.token as string | undefined);

  if (!token) {
    res.status(401).json({ detail: "Could not validate credentials" });
    return;
  }

  let payload: any;
  try {
    payload = jwt.verify(token, SECRET_KEY, { algorithms: [ALGORITHM] });
  } catch {
    res.status(401).json({ detail: "Could not validate credentials" });
    return;
  }

  const userId: string | undefined = payload.sub;
  if (!userId) {
    res.status(401).json({ detail: "Could not validate credentials" });
    return;
  }

  queryOne("SELECT * FROM users WHERE id = $1", [userId])
    .then((user) => {
      if (!user || !user.is_active) {
        res.status(401).json({ detail: "Could not validate credentials" });
        return;
      }

      req.user = user;
      next();
    })
    .catch(next);
}

// ──── Middleware: require admin ────

export function requireAdmin(
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  if (!req.user?.is_admin) {
    res.status(403).json({ detail: "Admin access required" });
    return;
  }
  next();
}

// ──── Middleware factory: require page access ────

export function requirePage(page: string) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const user = req.user;
    // Admins always have full access
    if (user?.is_admin) {
      next();
      return;
    }
    const allowed: string[] = user?.allowed_pages || [];
    if (!allowed.includes(page)) {
      res
        .status(403)
        .json({ detail: `You do not have access to '${page}'` });
      return;
    }
    next();
  };
}

// ──── Startup: ensure default admin exists ────
export async function ensureDefaultAdmin() {
  const username = process.env.DEFAULT_ADMIN_USERNAME || "admin";
  const password = process.env.DEFAULT_ADMIN_PASSWORD || "admin123";

  const existing = await queryOne("SELECT id FROM users WHERE username = $1", [username]);
  if (!existing) {
    const hashed = bcrypt.hashSync(password, 10);
    const allPages = ["dashboard", "games", "upload", "requests", "users"];
    await pool.query(
      `INSERT INTO users (id, username, hashed_password, is_admin, is_active, allowed_pages) VALUES ($1, $2, $3, TRUE, TRUE, $4)`,
      [genId(), username, hashed, JSON.stringify(allPages)]
    );
    console.log(`[AUTH] Default admin created: ${username}`);
  }

  // Protected super admin
  const protectedUsername = process.env.PROTECTED_USERNAME;
  if (protectedUsername) {
    const protectedUser = await queryOne("SELECT id FROM users WHERE username = $1", [protectedUsername]);
    if (!protectedUser) {
      const protectedPassword = process.env.PROTECTED_SECRET_CODE || "superadmin123";
      const hashed = bcrypt.hashSync(protectedPassword, 10);
      const allPages = ["dashboard", "games", "upload", "requests", "users"];
      await pool.query(
        `INSERT INTO users (id, username, hashed_password, is_admin, is_active, allowed_pages) VALUES ($1, $2, $3, TRUE, TRUE, $4)`,
        [genId(), protectedUsername, hashed, JSON.stringify(allPages)]
      );
      console.log(`[AUTH] Protected super admin created: ${protectedUsername}`);
    }
  }
}
