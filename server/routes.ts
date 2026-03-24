import type { Express } from "express";
import { createServer, type Server } from "http";
import { errorHandler } from "./middleware/errorHandler.js";
import { initDB } from "./db.js";

// Import route modules
import authRouter from "./routes/auth.js";
import gamesRouter from "./routes/games.js";
import playerRouter from "./routes/player.js";
import analyticsRouter from "./routes/analytics.js";
import usersRouter from "./routes/users.js";
import requestsRouter from "./routes/requests.js";
import evalRouter from "./routes/eval.js";
import settingsRouter from "./routes/settings.js";

// Import startup helpers
import { ensureDefaultAdmin } from "./middleware/auth.js";

export async function registerRoutes(app: Express): Promise<Server> {
  // ──── Startup: init DB tables ────
  await initDB();

  // ──── Startup: ensure default admin exists ────
  await ensureDefaultAdmin();

  // ──── Health check ────
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", app: "ARC-AGI Internal", timestamp: new Date().toISOString() });
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

  // ──── Error handler (must be last) ────
  app.use(errorHandler);

  return createServer(app);
}
