import { Router } from "express";
import multer from "multer";
import fs from "fs";
import os from "os";
import path from "path";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken } from "../middleware/auth.js";
import { AppError } from "../middleware/errorHandler.js";
import pool, { genId, queryOne, queryAll, toJsonb } from "../db.js";
import { GamePythonBridge, type GameFrame } from "../services/GamePythonBridge.js";

const router = Router();
const upload = multer({ storage: multer.memoryStorage() });
const ENV_DIR = process.env.ENVIRONMENT_FILES_DIR || path.join(process.cwd(), "environment_files");

// ──────────────────────────────────────────────
// In-memory store for active game bridges
// ──────────────────────────────────────────────

const activeBridges = new Map<string, GamePythonBridge>();
const tempDirs = new Map<string, string>();

/** Metadata tracked alongside each bridge for DB updates. */
interface BridgeMeta {
  gameDbId: string;     // games.id (PK) for stat updates
  gameId: string;       // games.game_id (human readable)
  isEphemeral: boolean;
}

const bridgeMeta = new Map<string, BridgeMeta>();

/** Per-session timing / level tracking (replaces Python GameEngineService tracking). */
interface SessionTracker {
  startTime: number;
  levelStartTime: number;
  levelStartActions: number;
  currentLevel: number;
  completedLevels: Array<{
    level: number;
    actions: number;
    time: number;
    completed: boolean;
    game_overs: number;
    resets: number;
  }>;
  actionsSinceReset: number;
  // Per-level counters (current level)
  levelGameOvers: number;    // GAME_OVER events on current level
  levelResets: number;       // manual R resets on current level
  // Session-wide counters
  totalGameOvers: number;
  totalResets: number;
  prevState: string;         // to detect GAME_OVER transitions
}

const sessionTrackers = new Map<string, SessionTracker>();

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

/** Build the canonical game .py path from the local_dir and game_code. */
function resolveGamePath(localDir: string, gameCode: string): string {
  return path.join(localDir, `${gameCode}.py`);
}

/**
 * Format a GameFrame into the JSON shape returned to the client.
 * Matches the Python GameFrameResponse schema.
 */
function formatFrameResponse(frame: GameFrame, sessionGuid: string, extra?: Record<string, any>) {
  const tracker = sessionTrackers.get(sessionGuid);
  const now = Date.now();

  const totalTime = tracker ? (now - tracker.startTime) / 1000 : 0;
  const currentLevelStats = tracker ? {
    level: tracker.currentLevel,
    actions: (frame.total_actions || 0) - tracker.levelStartActions,
    time: Math.round(((now - tracker.levelStartTime) / 1000) * 100) / 100,
    completed: false,
    game_overs: tracker.levelGameOvers,
    resets: tracker.levelResets,
  } : null;

  return {
    grid: frame.grid,
    width: frame.width,
    height: frame.height,
    state: frame.state,
    score: 0,
    level: frame.level,
    total_actions: frame.total_actions,
    available_actions: frame.available_actions,
    metadata: {
      session_guid: sessionGuid,
      total_time: Math.round(totalTime * 100) / 100,
      completed_levels: tracker?.completedLevels || [],
      current_level_stats: currentLevelStats,
      game_id: frame.game_id,
      game_overs: tracker?.totalGameOvers || 0,
      total_resets: tracker?.totalResets || 0,
      ...(extra || {}),
    },
  };
}

/**
 * After an action, persist frame state to the play_sessions table.
 * If the game ended (WIN / GAME_OVER), also update game-level stats.
 */
async function updatePlaySessionFromFrame(sessionGuid: string, frame: GameFrame, meta: BridgeMeta) {
  const tracker = sessionTrackers.get(sessionGuid);
  const now = Date.now();
  const totalTime = tracker ? Math.round(((now - tracker.startTime) / 1000) * 100) / 100 : 0;
  const completedLevels = tracker?.completedLevels || [];
  const totalGameOvers = tracker?.totalGameOvers || 0;

  await pool.query(
    `UPDATE play_sessions
        SET state = $1, total_actions = $2, current_level = $3,
            total_time = $4, level_stats = $5, game_overs = $6
      WHERE session_guid = $7`,
    [
      frame.state,
      frame.total_actions,
      frame.level,
      totalTime,
      toJsonb(completedLevels),
      totalGameOvers,
      sessionGuid,
    ],
  );

  // On terminal state, record end time and bump game-wide counters
  if (frame.state === "WIN" || frame.state === "GAME_OVER") {
    await pool.query(
      `UPDATE play_sessions SET ended_at = NOW() WHERE session_guid = $1`,
      [sessionGuid],
    );

    await pool.query(`UPDATE games SET total_plays = COALESCE(total_plays, 0) + 1 WHERE id = $1`, [meta.gameDbId]);

    if (frame.state === "WIN") {
      await pool.query(`UPDATE games SET total_wins = COALESCE(total_wins, 0) + 1 WHERE id = $1`, [meta.gameDbId]);
    }
  }
}

/**
 * After an action in an ephemeral session, persist to temp_game_sessions table.
 */
async function updateTempSessionFromFrame(
  sessionGuid: string,
  frame: GameFrame,
  action: string,
) {
  const tracker = sessionTrackers.get(sessionGuid);
  const now = Date.now();
  const totalTime = tracker ? Math.round(((now - tracker.startTime) / 1000) * 100) / 100 : 0;
  const completedLevels = tracker?.completedLevels || [];
  const totalGameOvers = tracker?.totalGameOvers || 0;

  await pool.query(
    `UPDATE temp_game_sessions
        SET state = $1, total_actions = $2, current_level = $3,
            total_time = $4, level_stats = $5, score = $6, game_overs = $7
      WHERE session_guid = $8`,
    [
      frame.state,
      frame.total_actions,
      frame.level,
      totalTime,
      toJsonb(completedLevels),
      0,
      totalGameOvers,
      sessionGuid,
    ],
  );

  // Append to action_log
  const row = await queryOne(
    `SELECT action_log FROM temp_game_sessions WHERE session_guid = $1`,
    [sessionGuid],
  );

  if (row) {
    const log: any[] = row.action_log || [];
    log.push({ action, level: frame.level, time: totalTime });
    await pool.query(`UPDATE temp_game_sessions SET action_log = $1 WHERE session_guid = $2`, [
      toJsonb(log),
      sessionGuid,
    ]);
  }
}

// ──────────────────────────────────────────────
// ARC Color Palette constant
// ──────────────────────────────────────────────

const ARC_PALETTE: Record<number, string> = {
  0: "#000000",
  1: "#0074D9",
  2: "#FF4136",
  3: "#2ECC40",
  4: "#FFDC00",
  5: "#AAAAAA",
  6: "#F012BE",
  7: "#FF851B",
  8: "#7FDBFF",
  9: "#870C25",
};

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENDPOINTS (no auth) — 3 endpoints
// ════════════════════════════════════════════════════════════════════════════

// ──── 1. POST /public/start — Start public guest session ────
router.post(
  "/public/start",
  asyncHandler(async (req, res) => {
    const { game_id, seed, player_name, start_level } = req.body;

    if (!game_id) {
      throw new AppError("game_id is required", 400);
    }

    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [game_id],
    );

    if (!game) {
      throw new AppError("Game not found or inactive", 404);
    }

    const sessionGuid = crypto.randomUUID();
    const effectiveSeed = seed ?? 0;
    const gamePath = resolveGamePath(game.local_dir, game.game_code);

    // Create Python bridge and initialise
    const bridge = new GamePythonBridge();
    try {
      await bridge.init(game.game_id, gamePath, effectiveSeed);

      // Skip to requested start level
      if (start_level && start_level > 0) {
        for (let i = 0; i < start_level; i++) {
          await bridge.sendAction("noop");
        }
      }
    } catch (e: any) {
      bridge.kill();
      throw new AppError(`Failed to start game: ${e.message}`, 500);
    }

    // Store bridge
    activeBridges.set(sessionGuid, bridge);
    bridgeMeta.set(sessionGuid, {
      gameDbId: game.id,
      gameId: game.game_id,
      isEphemeral: false,
    });

    // Create play session in DB
    const sessionId = genId();
    await pool.query(
      `INSERT INTO play_sessions (id, game_id, user_id, session_guid, seed, player_name, started_at)
       VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
      [sessionId, game.id, null, sessionGuid, effectiveSeed, player_name || null],
    );

    // Get initial frame by sending a no-op reset to get current state
    const frame = await bridge.reset();

    // Initialise session tracker
    const now = Date.now();
    sessionTrackers.set(sessionGuid, {
      startTime: now,
      levelStartTime: now,
      levelStartActions: 0,
      currentLevel: frame.level || 0,
      completedLevels: [],
      actionsSinceReset: 0,
      levelGameOvers: 0, levelResets: 0,
      totalGameOvers: 0, totalResets: 0,
      prevState: frame.state || 'NOT_FINISHED',
    });

    res.json(formatFrameResponse(frame, sessionGuid));
  }),
);

// ──── 2. POST /public/action/:sessionGuid — Execute action ────
router.post(
  "/public/action/:sessionGuid",
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;
    const { action, x, y } = req.body;

    if (!action) {
      throw new AppError("action is required", 400);
    }

    // Check if session is already won in DB
    const playSession = await queryOne(
      "SELECT * FROM play_sessions WHERE session_guid = $1",
      [sessionGuid],
    );

    if (!playSession) {
      throw new AppError("Play session not found", 404);
    }

    const bridge = activeBridges.get(sessionGuid);

    // If already won, return cached final state
    if (playSession.state === "WIN") {
      if (bridge) {
        try {
          const frame = await bridge.reset(); // reset returns current frame
          res.json(formatFrameResponse(frame, sessionGuid));
          return;
        } catch {
          // fallback to DB state
        }
      }
      res.json({
        grid: [[0]],
        width: 1,
        height: 1,
        state: playSession.state,
        score: playSession.score || 0,
        level: playSession.current_level || 0,
        total_actions: playSession.total_actions || 0,
        available_actions: [],
        metadata: { session_guid: sessionGuid },
      });
      return;
    }

    if (!bridge) {
      throw new AppError("Game session not found or expired", 404);
    }

    // Handle RESET action — reset level timer
    const tracker = sessionTrackers.get(sessionGuid);
    if (action === "RESET" && tracker) {
      tracker.levelResets++;
      tracker.totalResets++;
      if (tracker.actionsSinceReset === 0) {
        // Double reset = restart from level 1
        tracker.completedLevels = [];
        tracker.currentLevel = 0;
        tracker.startTime = Date.now();
        tracker.levelGameOvers = 0;
        tracker.levelResets = 0;
      }
      tracker.actionsSinceReset = 0;
      tracker.levelStartTime = Date.now();
      tracker.levelStartActions = 0; // will be updated after frame comes back
    }

    // Execute action
    let frame: GameFrame;
    try {
      frame = await bridge.sendAction(action, x, y);
    } catch (e: any) {
      throw new AppError(`Action failed: ${e.message}`, 500);
    }

    // Track level transitions + game overs
    if (tracker) {
      const now = Date.now();
      tracker.actionsSinceReset++;

      // Detect GAME_OVER transition (= lost a life on current level)
      if (frame.state === "GAME_OVER" && tracker.prevState !== "GAME_OVER") {
        tracker.levelGameOvers++;
        tracker.totalGameOvers++;
      }
      tracker.prevState = frame.state || "NOT_FINISHED";

      // Detect level change
      const newLevel = frame.level ?? 0;
      if (newLevel > tracker.currentLevel) {
        // Level was completed — record stats including lives + resets
        const levelTime = (now - tracker.levelStartTime) / 1000;
        const levelActions = (frame.total_actions || 0) - tracker.levelStartActions;
        tracker.completedLevels.push({
          level: tracker.currentLevel,
          actions: levelActions,
          time: Math.round(levelTime * 100) / 100,
          completed: true,
          game_overs: tracker.levelGameOvers,
          resets: tracker.levelResets,
        });
        // Reset per-level counters for new level
        tracker.levelStartTime = now;
        tracker.levelStartActions = frame.total_actions || 0;
        tracker.currentLevel = newLevel;
        tracker.levelGameOvers = 0;
        tracker.levelResets = 0;
      }
    }

    // Persist to DB
    const meta = bridgeMeta.get(sessionGuid);
    if (meta) {
      await updatePlaySessionFromFrame(sessionGuid, frame, meta);
    }

    res.json(formatFrameResponse(frame, sessionGuid));
  }),
);

// ──── 3. POST /public/end/:sessionGuid — End session ────
router.post(
  "/public/end/:sessionGuid",
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;

    // Update DB if session is unfinished
    const playSession = await queryOne(
      "SELECT * FROM play_sessions WHERE session_guid = $1",
      [sessionGuid],
    );

    if (playSession && playSession.state === "NOT_FINISHED") {
      const bridge = activeBridges.get(sessionGuid);
      const meta = bridgeMeta.get(sessionGuid);

      // Try to capture final timing from bridge
      if (bridge) {
        try {
          const frame = await bridge.reset();
          const totalTime = (frame as any).metadata?.total_time
            || (frame as any).total_time
            || 0;
          const completedLevels = (frame as any).metadata?.completed_levels
            || (frame as any).completed_levels
            || [];
          await pool.query(
            `UPDATE play_sessions SET total_time = $1, level_stats = $2 WHERE session_guid = $3`,
            [totalTime, toJsonb(completedLevels), sessionGuid],
          );
        } catch {
          // ignore — engine may already be dead
        }
      }

      await pool.query(
        `UPDATE play_sessions SET state = 'GAME_OVER', ended_at = NOW() WHERE session_guid = $1`,
        [sessionGuid],
      );

      // Increment total_plays on the game
      if (meta) {
        await pool.query(`UPDATE games SET total_plays = COALESCE(total_plays, 0) + 1 WHERE id = $1`, [meta.gameDbId]);
      }
    }

    // Kill bridge
    const bridge = activeBridges.get(sessionGuid);
    if (bridge) {
      bridge.kill();
      activeBridges.delete(sessionGuid);
    }
    bridgeMeta.delete(sessionGuid);
    sessionTrackers.delete(sessionGuid);

    res.json({ detail: "Session ended" });
  }),
);

// ════════════════════════════════════════════════════════════════════════════
// EPHEMERAL ENDPOINTS (no auth, temp files) — 3 endpoints
// ════════════════════════════════════════════════════════════════════════════

// ──── 4. POST /ephemeral/start — Upload game.py + metadata.json, start temp session ────
router.post(
  "/ephemeral/start",
  upload.fields([
    { name: "game_file", maxCount: 1 },
    { name: "metadata_file", maxCount: 1 },
  ]),
  asyncHandler(async (req, res) => {
    const gameFile = (req.files as any)?.game_file?.[0];
    const metadataFile = (req.files as any)?.metadata_file?.[0];

    if (!gameFile) {
      throw new AppError("game_file is required", 400);
    }
    if (!metadataFile) {
      throw new AppError("metadata_file is required", 400);
    }

    const gamePyBytes: Buffer = gameFile.buffer;
    const metadataBytes: Buffer = metadataFile.buffer;

    // Parse metadata
    let metadata: any;
    try {
      metadata = JSON.parse(metadataBytes.toString("utf-8"));
    } catch {
      throw new AppError("Invalid metadata.json", 400);
    }

    const gameId: string = metadata.game_id || "temp";
    const playerName = ((req.body.player_name as string) || "").trim();

    // Derive game_code from game_id
    let gameCode: string;
    if (gameId.includes("-")) {
      gameCode = gameId.substring(0, gameId.lastIndexOf("-"));
    } else {
      gameCode = gameId;
    }

    // Write uploaded files to a temp directory
    const tempBase = fs.mkdtempSync(path.join(os.tmpdir(), "arc_ephemeral_"));
    const gameDir = path.join(tempBase, gameCode, "v1");
    fs.mkdirSync(gameDir, { recursive: true });

    const gamePyPath = path.join(gameDir, `${gameCode}.py`);
    const metadataPath = path.join(gameDir, "metadata.json");

    fs.writeFileSync(gamePyPath, gamePyBytes);
    fs.writeFileSync(metadataPath, metadataBytes);

    // Create session guid with ephemeral prefix
    const sessionGuid = `eph_${crypto.randomUUID().replace(/-/g, "").substring(0, 12)}`;

    // Create Python bridge
    const bridge = new GamePythonBridge();
    try {
      await bridge.init(gameId, gamePyPath, 0);
    } catch (e: any) {
      bridge.kill();
      fs.rmSync(tempBase, { recursive: true, force: true });
      throw new AppError(`Failed to start game: ${e.message}`, 500);
    }

    // Store bridge and temp dir
    activeBridges.set(sessionGuid, bridge);
    tempDirs.set(sessionGuid, tempBase);
    bridgeMeta.set(sessionGuid, {
      gameDbId: "",       // ephemeral — no games row
      gameId,
      isEphemeral: true,
    });

    // Log to temp_game_sessions table
    await pool.query(
      `INSERT INTO temp_game_sessions (id, session_guid, game_id, player_name, state, started_at, expires_at, action_log)
       VALUES ($1, $2, $3, $4, 'NOT_FINISHED', NOW(), NOW() + INTERVAL '1 day', $5)`,
      [genId(), sessionGuid, gameId, playerName || null, toJsonb([])],
    );

    // Get initial frame
    const frame = await bridge.reset();

    // Initialise session tracker
    const now = Date.now();
    sessionTrackers.set(sessionGuid, {
      startTime: now,
      levelStartTime: now,
      levelStartActions: 0,
      currentLevel: frame.level || 0,
      completedLevels: [],
      actionsSinceReset: 0,
      levelGameOvers: 0, levelResets: 0,
      totalGameOvers: 0, totalResets: 0,
      prevState: frame.state || 'NOT_FINISHED',
    });

    res.json(formatFrameResponse(frame, sessionGuid, { ephemeral: true }));
  }),
);

// ──── 5. POST /ephemeral/action/:sessionGuid — Execute action ────
router.post(
  "/ephemeral/action/:sessionGuid",
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;
    const { action, x, y } = req.body;

    if (!action) {
      throw new AppError("action is required", 400);
    }

    const bridge = activeBridges.get(sessionGuid);
    if (!bridge) {
      throw new AppError("Ephemeral session not found or expired", 404);
    }

    // Handle RESET action — reset level timer
    const tracker = sessionTrackers.get(sessionGuid);
    if (action === "RESET" && tracker) {
      tracker.levelResets++;
      tracker.totalResets++;
      if (tracker.actionsSinceReset === 0) {
        // Double reset = restart from level 1
        tracker.completedLevels = [];
        tracker.currentLevel = 0;
        tracker.startTime = Date.now();
        tracker.levelGameOvers = 0;
        tracker.levelResets = 0;
      }
      tracker.actionsSinceReset = 0;
      tracker.levelStartTime = Date.now();
      tracker.levelStartActions = 0; // will be updated after frame comes back
    }

    let frame: GameFrame;
    try {
      frame = await bridge.sendAction(action, x, y);
    } catch (e: any) {
      throw new AppError(`Action failed: ${e.message}`, 500);
    }

    // Track level transitions + game overs
    if (tracker) {
      const now = Date.now();
      tracker.actionsSinceReset++;

      // Detect GAME_OVER transition (= lost a life on current level)
      if (frame.state === "GAME_OVER" && tracker.prevState !== "GAME_OVER") {
        tracker.levelGameOvers++;
        tracker.totalGameOvers++;
      }
      tracker.prevState = frame.state || "NOT_FINISHED";

      // Detect level change
      const newLevel = frame.level ?? 0;
      if (newLevel > tracker.currentLevel) {
        // Level was completed — record stats including lives + resets
        const levelTime = (now - tracker.levelStartTime) / 1000;
        const levelActions = (frame.total_actions || 0) - tracker.levelStartActions;
        tracker.completedLevels.push({
          level: tracker.currentLevel,
          actions: levelActions,
          time: Math.round(levelTime * 100) / 100,
          completed: true,
          game_overs: tracker.levelGameOvers,
          resets: tracker.levelResets,
        });
        // Reset per-level counters for new level
        tracker.levelStartTime = now;
        tracker.levelStartActions = frame.total_actions || 0;
        tracker.currentLevel = newLevel;
        tracker.levelGameOvers = 0;
        tracker.levelResets = 0;
      }
    }

    // Persist to temp_game_sessions
    await updateTempSessionFromFrame(sessionGuid, frame, action);

    res.json(formatFrameResponse(frame, sessionGuid, { ephemeral: true }));
  }),
);

// ──── 6. POST /ephemeral/end/:sessionGuid — End session + cleanup ────
router.post(
  "/ephemeral/end/:sessionGuid",
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;

    // Finalise temp session in DB
    const tempSession = await queryOne(
      "SELECT * FROM temp_game_sessions WHERE session_guid = $1",
      [sessionGuid],
    );

    if (tempSession && tempSession.state === "NOT_FINISHED") {
      const bridge = activeBridges.get(sessionGuid);

      // Try to capture final state before destroying
      if (bridge) {
        try {
          const frame = await bridge.reset();
          const totalTime = (frame as any).metadata?.total_time
            || (frame as any).total_time
            || 0;
          const completedLevels = (frame as any).metadata?.completed_levels
            || (frame as any).completed_levels
            || [];
          const totalGameOvers = (frame as any).metadata?.total_game_overs
            ?? (frame as any).total_game_overs;

          await pool.query(
            `UPDATE temp_game_sessions SET total_time = $1, level_stats = $2 WHERE session_guid = $3`,
            [totalTime, toJsonb(completedLevels), sessionGuid],
          );

          if (totalGameOvers !== undefined && totalGameOvers !== null) {
            await pool.query(
              `UPDATE temp_game_sessions SET game_overs = $1 WHERE session_guid = $2`,
              [totalGameOvers, sessionGuid],
            );
          }
        } catch {
          // engine may already be dead
        }
      }

      await pool.query(
        `UPDATE temp_game_sessions SET state = 'GAME_OVER', ended_at = NOW() WHERE session_guid = $1`,
        [sessionGuid],
      );
    }

    // Kill bridge
    const bridge = activeBridges.get(sessionGuid);
    if (bridge) {
      bridge.kill();
      activeBridges.delete(sessionGuid);
    }
    bridgeMeta.delete(sessionGuid);
    sessionTrackers.delete(sessionGuid);

    // Clean up temp directory
    const tempDir = tempDirs.get(sessionGuid);
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
    tempDirs.delete(sessionGuid);

    res.json({ detail: "Ephemeral session ended" });
  }),
);

// ════════════════════════════════════════════════════════════════════════════
// ADMIN ENDPOINTS (auth required) — 4 endpoints
// ════════════════════════════════════════════════════════════════════════════

// ──── 7. POST /start — Start authenticated session ────
router.post(
  "/start",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { game_id, seed, start_level } = req.body;
    const userId = req.user.id;

    if (!game_id) {
      throw new AppError("game_id is required", 400);
    }

    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1",
      [game_id],
    );

    if (!game) {
      throw new AppError("Game not found", 404);
    }
    if (!game.is_active) {
      throw new AppError("Game is deactivated", 400);
    }

    const sessionGuid = crypto.randomUUID();
    const effectiveSeed = seed ?? 0;
    const gamePath = resolveGamePath(game.local_dir, game.game_code);

    // Create Python bridge
    const bridge = new GamePythonBridge();
    try {
      await bridge.init(game.game_id, gamePath, effectiveSeed);

      // Skip to requested level
      if (start_level && start_level > 0) {
        for (let i = 0; i < start_level; i++) {
          await bridge.sendAction("noop");
        }
      }
    } catch (e: any) {
      bridge.kill();
      throw new AppError(`Failed to start game: ${e.message}`, 500);
    }

    // Store bridge
    activeBridges.set(sessionGuid, bridge);
    bridgeMeta.set(sessionGuid, {
      gameDbId: game.id,
      gameId: game.game_id,
      isEphemeral: false,
    });

    // Insert play session into DB
    const sessionId = genId();
    await pool.query(
      `INSERT INTO play_sessions (id, game_id, user_id, session_guid, seed, started_at)
       VALUES ($1, $2, $3, $4, $5, NOW())`,
      [sessionId, game.id, userId, sessionGuid, effectiveSeed],
    );

    // Get initial frame
    const frame = await bridge.reset();

    // Initialise session tracker
    const now = Date.now();
    sessionTrackers.set(sessionGuid, {
      startTime: now,
      levelStartTime: now,
      levelStartActions: 0,
      currentLevel: frame.level || 0,
      completedLevels: [],
      actionsSinceReset: 0,
      levelGameOvers: 0, levelResets: 0,
      totalGameOvers: 0, totalResets: 0,
      prevState: frame.state || 'NOT_FINISHED',
    });

    res.json(formatFrameResponse(frame, sessionGuid));
  }),
);

// ──── 8. POST /action/:sessionGuid — Execute action (authenticated) ────
router.post(
  "/action/:sessionGuid",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;
    const { action, x, y } = req.body;

    if (!action) {
      throw new AppError("action is required", 400);
    }

    // Verify session exists
    const playSession = await queryOne(
      "SELECT * FROM play_sessions WHERE session_guid = $1",
      [sessionGuid],
    );

    if (!playSession) {
      throw new AppError("Play session not found", 404);
    }

    const bridge = activeBridges.get(sessionGuid);

    // If already won, return cached/final state
    if (playSession.state === "WIN") {
      if (bridge) {
        try {
          const frame = await bridge.reset();
          res.json(formatFrameResponse(frame, sessionGuid));
          return;
        } catch {
          // fallback to DB state
        }
      }
      res.json({
        grid: [[0]],
        width: 1,
        height: 1,
        state: playSession.state,
        score: playSession.score || 0,
        level: playSession.current_level || 0,
        total_actions: playSession.total_actions || 0,
        available_actions: [],
        metadata: { session_guid: sessionGuid },
      });
      return;
    }

    if (!bridge) {
      throw new AppError("Game session not found or expired", 404);
    }

    // Handle RESET action — reset level timer
    const tracker = sessionTrackers.get(sessionGuid);
    if (action === "RESET" && tracker) {
      tracker.levelResets++;
      tracker.totalResets++;
      if (tracker.actionsSinceReset === 0) {
        // Double reset = restart from level 1
        tracker.completedLevels = [];
        tracker.currentLevel = 0;
        tracker.startTime = Date.now();
        tracker.levelGameOvers = 0;
        tracker.levelResets = 0;
      }
      tracker.actionsSinceReset = 0;
      tracker.levelStartTime = Date.now();
      tracker.levelStartActions = 0; // will be updated after frame comes back
    }

    // Execute action
    let frame: GameFrame;
    try {
      frame = await bridge.sendAction(action, x, y);
    } catch (e: any) {
      throw new AppError(`Action failed: ${e.message}`, 500);
    }

    // Track level transitions + game overs
    if (tracker) {
      const now = Date.now();
      tracker.actionsSinceReset++;

      // Detect GAME_OVER transition (= lost a life on current level)
      if (frame.state === "GAME_OVER" && tracker.prevState !== "GAME_OVER") {
        tracker.levelGameOvers++;
        tracker.totalGameOvers++;
      }
      tracker.prevState = frame.state || "NOT_FINISHED";

      // Detect level change
      const newLevel = frame.level ?? 0;
      if (newLevel > tracker.currentLevel) {
        // Level was completed — record stats including lives + resets
        const levelTime = (now - tracker.levelStartTime) / 1000;
        const levelActions = (frame.total_actions || 0) - tracker.levelStartActions;
        tracker.completedLevels.push({
          level: tracker.currentLevel,
          actions: levelActions,
          time: Math.round(levelTime * 100) / 100,
          completed: true,
          game_overs: tracker.levelGameOvers,
          resets: tracker.levelResets,
        });
        // Reset per-level counters for new level
        tracker.levelStartTime = now;
        tracker.levelStartActions = frame.total_actions || 0;
        tracker.currentLevel = newLevel;
        tracker.levelGameOvers = 0;
        tracker.levelResets = 0;
      }
    }

    // Persist to DB
    const meta = bridgeMeta.get(sessionGuid);
    if (meta) {
      await updatePlaySessionFromFrame(sessionGuid, frame, meta);
    }

    res.json(formatFrameResponse(frame, sessionGuid));
  }),
);

// ──── 9. GET /frame/:sessionGuid — Get current frame (authenticated) ────
router.get(
  "/frame/:sessionGuid",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;

    const bridge = activeBridges.get(sessionGuid);
    if (!bridge) {
      throw new AppError("Game session not found or expired", 404);
    }

    let frame: GameFrame;
    try {
      frame = await bridge.reset();
    } catch (e: any) {
      throw new AppError(`Failed to get frame: ${e.message}`, 500);
    }

    res.json(formatFrameResponse(frame, sessionGuid));
  }),
);

// ──── 10. POST /end/:sessionGuid — End session (authenticated) ────
router.post(
  "/end/:sessionGuid",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { sessionGuid } = req.params;

    // Update DB if session is unfinished
    const playSession = await queryOne(
      "SELECT * FROM play_sessions WHERE session_guid = $1",
      [sessionGuid],
    );

    if (playSession && playSession.state === "NOT_FINISHED") {
      const bridge = activeBridges.get(sessionGuid);
      const meta = bridgeMeta.get(sessionGuid);

      // Try to capture final timing
      if (bridge) {
        try {
          const frame = await bridge.reset();
          const totalTime = (frame as any).metadata?.total_time
            || (frame as any).total_time
            || 0;
          const completedLevels = (frame as any).metadata?.completed_levels
            || (frame as any).completed_levels
            || [];
          await pool.query(
            `UPDATE play_sessions SET total_time = $1, level_stats = $2 WHERE session_guid = $3`,
            [totalTime, toJsonb(completedLevels), sessionGuid],
          );
        } catch {
          // ignore — engine may already be dead
        }
      }

      await pool.query(
        `UPDATE play_sessions SET state = 'GAME_OVER', ended_at = NOW() WHERE session_guid = $1`,
        [sessionGuid],
      );

      // Increment total_plays on the game
      if (meta) {
        await pool.query(`UPDATE games SET total_plays = COALESCE(total_plays, 0) + 1 WHERE id = $1`, [meta.gameDbId]);
      }
    }

    // Kill bridge
    const bridge = activeBridges.get(sessionGuid);
    if (bridge) {
      bridge.kill();
      activeBridges.delete(sessionGuid);
    }
    bridgeMeta.delete(sessionGuid);
    sessionTrackers.delete(sessionGuid);

    res.json({ detail: "Session ended" });
  }),
);

// ════════════════════════════════════════════════════════════════════════════
// PALETTE (no auth) — 1 endpoint
// ════════════════════════════════════════════════════════════════════════════

// ──── 11. GET /palette — Return ARC color palette ────
router.get(
  "/palette",
  asyncHandler(async (_req, res) => {
    res.json(ARC_PALETTE);
  }),
);

export default router;
