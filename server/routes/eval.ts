import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireAdmin } from "../middleware/auth.js";
import { AppError } from "../middleware/errorHandler.js";
import { queryAll } from "../db.js";
import { AVAILABLE_MODELS } from "../services/eval/LLMProvider.js";
import { startEval, getEvalSession, cancelEval, getAllSessions, type EvalConfig } from "../services/eval/EvalRunner.js";
import { sseManager } from "../services/eval/SSEStreamManager.js";
import crypto from "crypto";

const router = Router();

// ──── 1. GET /models — List available models ────
router.get("/models", (_req, res) => {
  res.json(AVAILABLE_MODELS);
});

// ──── 2. POST /start — Start an evaluation ────
router.post(
  "/start",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { games, models, runsPerGame, maxSteps, contextWindow, systemPrompt, reasoningEffort } = req.body;

    if (!Array.isArray(games) || games.length === 0) {
      throw new AppError("games must be a non-empty array of game IDs", 400);
    }
    if (!Array.isArray(models) || models.length === 0) {
      throw new AppError("models must be a non-empty array of model IDs", 400);
    }

    // Validate model IDs exist in AVAILABLE_MODELS
    const availableIds = AVAILABLE_MODELS.map((m: any) => m.id);
    const invalidModels = models.filter((id: string) => !availableIds.includes(id));
    if (invalidModels.length > 0) {
      throw new AppError(`Unknown model(s): ${invalidModels.join(", ")}`, 400);
    }

    // Validate games exist in DB
    const placeholders = games.map((_: string, i: number) => `$${i + 1}`).join(", ");
    const gameRows = await queryAll(
      `SELECT id, game_id, game_code, local_dir FROM games WHERE game_id IN (${placeholders})`,
      games,
    );

    const foundIds = new Set(gameRows.map((r: any) => r.game_id));
    const missingGames = games.filter((g: string) => !foundIds.has(g));
    if (missingGames.length > 0) {
      throw new AppError(`Games not found: ${missingGames.join(", ")}`, 404);
    }

    // Build config
    const evalId = crypto.randomUUID();
    const config: EvalConfig = {
      evalId,
      games: gameRows.map((r: any) => ({
        gameId: r.game_id,
        gameCode: r.game_code,
        localDir: r.local_dir,
      })),
      models,
      runsPerGame: runsPerGame ?? 1,
      maxSteps: maxSteps ?? 200,
      contextWindow: contextWindow ?? 10,
      systemPrompt: systemPrompt ?? undefined,
      reasoningEffort: reasoningEffort ?? "medium",
    };

    // Fire-and-forget — let the eval run in the background
    startEval(config);

    res.json({ evalId });
  }),
);

// ──── 3. GET /:evalId/stream — SSE stream for eval events ────
router.get(
  "/:evalId/stream",
  asyncHandler(async (req, res) => {
    const { evalId } = req.params;

    const session = getEvalSession(evalId);
    if (!session) {
      throw new AppError("Eval session not found", 404);
    }

    sseManager.register(evalId, res);
  }),
);

// ──── 4. POST /:evalId/cancel — Cancel a running eval ────
router.post(
  "/:evalId/cancel",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { evalId } = req.params;

    cancelEval(evalId);

    res.json({ ok: true });
  }),
);

// ──── 5. GET /:evalId/status — Get eval session status ────
router.get(
  "/:evalId/status",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { evalId } = req.params;

    const session = getEvalSession(evalId);
    if (!session) {
      throw new AppError("Eval session not found", 404);
    }

    res.json(session);
  }),
);

// ──── 6. GET /sessions — List recent eval sessions ────
router.get(
  "/sessions",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (_req, res) => {
    const sessions = getAllSessions();
    res.json(sessions);
  }),
);

export default router;
