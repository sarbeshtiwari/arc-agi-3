import { Router } from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireAdmin } from "../middleware/auth.js";
import { AppError } from "../middleware/errorHandler.js";
import pool, { genId, queryOne, queryAll, toJsonb } from "../db.js";

const router = Router();
const upload = multer({ storage: multer.memoryStorage() });
const ENV_DIR = process.env.ENVIRONMENT_FILES_DIR || path.join(process.cwd(), "environment_files");

// ──── Allowed metadata keys (matches Python GameManagerService) ────

const ALLOWED_METADATA_KEYS = new Set([
  "game_id",
  "default_fps",
  "baseline_actions",
  "tags",
  "local_dir",
]);

// ──── Validation helpers (inlined from GameManagerService) ────

function validateMetadata(content: string): any {
  let metadata: any;
  try {
    metadata = JSON.parse(content);
  } catch (e: any) {
    throw new AppError(`Invalid JSON in metadata.json: ${e.message}`, 400);
  }

  if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) {
    throw new AppError("metadata.json must be a JSON object", 400);
  }

  if (!metadata.game_id) {
    throw new AppError("metadata.json must contain 'game_id' field", 400);
  }

  // Check for unknown keys
  const unknownKeys = Object.keys(metadata).filter((k) => !ALLOWED_METADATA_KEYS.has(k));
  if (unknownKeys.length > 0) {
    const allowed = Array.from(ALLOWED_METADATA_KEYS).sort().join(", ");
    throw new AppError(
      `metadata.json contains unknown fields: ${unknownKeys.sort().join(", ")}. Allowed fields: ${allowed}`,
      400,
    );
  }

  // Validate game_id format
  const gameId = metadata.game_id;
  if (typeof gameId !== "string" || !gameId) {
    throw new AppError("game_id must be a non-empty string", 400);
  }

  const basicPattern = /^[a-z]{2,6}\d{1,4}$/;
  const versionedPattern = /^[a-z0-9]{2,10}-v\d+$/;
  if (!basicPattern.test(gameId) && !versionedPattern.test(gameId)) {
    throw new AppError(
      `game_id '${gameId}' must match format '<code><number>' (e.g. 'ls20', 'ab12') ` +
        `or '<code>-v<number>' (e.g. 'ls20-v1')`,
      400,
    );
  }

  // Validate types
  if (metadata.default_fps !== undefined) {
    if (typeof metadata.default_fps !== "number" || !Number.isInteger(metadata.default_fps) || metadata.default_fps < 1) {
      throw new AppError("default_fps must be a positive integer", 400);
    }
  }

  if (metadata.baseline_actions !== undefined) {
    const ba = metadata.baseline_actions;
    if (!Array.isArray(ba) || !ba.every((x: any) => typeof x === "number" && Number.isInteger(x))) {
      throw new AppError("baseline_actions must be a list of integers", 400);
    }
  }

  if (metadata.tags !== undefined) {
    const tags = metadata.tags;
    if (!Array.isArray(tags) || !tags.every((t: any) => typeof t === "string")) {
      throw new AppError("tags must be a list of strings", 400);
    }
  }

  if (metadata.local_dir !== undefined) {
    if (typeof metadata.local_dir !== "string") {
      throw new AppError("local_dir must be a string", 400);
    }
  }

  return metadata;
}

/** Extract game_code and version from a full game_id. */
function parseGameId(fullGameId: string): { gameCode: string; version: string } {
  if (fullGameId.includes("-")) {
    const lastDash = fullGameId.lastIndexOf("-");
    return {
      gameCode: fullGameId.substring(0, lastDash),
      version: fullGameId.substring(lastDash + 1) || "v1",
    };
  }
  return { gameCode: fullGameId, version: "v1" };
}

/**
 * Upload game files to disk:
 * - Validates metadata + game file
 * - Writes to ENV_DIR/<game_code>/<version>/
 * - Returns result object with paths and parsed metadata
 */
function uploadGameFiles(gamePyContent: Buffer, metadataContent: Buffer): any {
  const metadataStr = metadataContent.toString("utf-8");
  const metadata = validateMetadata(metadataStr);

  const fullGameId: string = metadata.game_id;
  const { gameCode, version } = parseGameId(fullGameId);

  const gameDir = path.join(ENV_DIR, gameCode, version);
  fs.mkdirSync(gameDir, { recursive: true });

  const gamePyPath = path.join(gameDir, `${gameCode}.py`);
  const metadataPath = path.join(gameDir, "metadata.json");

  const gamePyStr = gamePyContent.toString("utf-8");
  fs.writeFileSync(gamePyPath, gamePyStr, "utf-8");

  // Update local_dir in metadata and write
  metadata.local_dir = path.join("environment_files", gameCode, version);
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), "utf-8");

  return {
    game_id: fullGameId,
    game_code: gameCode,
    version,
    local_dir: gameDir,
    game_file_path: gamePyPath,
    metadata_file_path: metadataPath,
    metadata,
  };
}

/** Delete game files from disk. */
function deleteGameFiles(gameCode: string, version: string): boolean {
  const gameDir = path.join(ENV_DIR, gameCode, version);
  if (fs.existsSync(gameDir)) {
    fs.rmSync(gameDir, { recursive: true, force: true });
    // Remove parent directory if empty
    const parent = path.join(ENV_DIR, gameCode);
    if (fs.existsSync(parent)) {
      try {
        const entries = fs.readdirSync(parent);
        if (entries.length === 0) fs.rmdirSync(parent);
      } catch { /* ignore */ }
    }
    return true;
  }
  return false;
}

/** Read game source file from disk. */
function getGameFileContent(gameCode: string, version: string): string | null {
  const gamePyPath = path.join(ENV_DIR, gameCode, version, `${gameCode}.py`);
  if (fs.existsSync(gamePyPath)) {
    return fs.readFileSync(gamePyPath, "utf-8");
  }
  return null;
}

/** Read metadata.json from disk. */
function getMetadataContent(gameCode: string, version: string): any | null {
  const metadataPath = path.join(ENV_DIR, gameCode, version, "metadata.json");
  if (fs.existsSync(metadataPath)) {
    try {
      return JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
    } catch {
      return null;
    }
  }
  return null;
}

/** Scan ENV_DIR for all local game directories with metadata.json. */
function listLocalGames(): any[] {
  const games: any[] = [];
  if (!fs.existsSync(ENV_DIR)) return games;

  for (const gameCodeDir of fs.readdirSync(ENV_DIR).sort()) {
    const gameCodePath = path.join(ENV_DIR, gameCodeDir);
    if (!fs.statSync(gameCodePath).isDirectory()) continue;

    for (const versionDir of fs.readdirSync(gameCodePath).sort()) {
      const versionPath = path.join(gameCodePath, versionDir);
      if (!fs.statSync(versionPath).isDirectory()) continue;

      const metadataPath = path.join(versionPath, "metadata.json");
      if (fs.existsSync(metadataPath)) {
        try {
          const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
          games.push({
            game_code: gameCodeDir,
            version: versionDir,
            game_id: metadata.game_id || `${gameCodeDir}-${versionDir}`,
            metadata,
            local_dir: versionPath,
          });
        } catch { /* skip broken metadata */ }
      }
    }
  }

  return games;
}

// ──── Format helpers ────

function formatGame(row: any): any {
  if (!row) return null;
  return {
    id: row.id,
    game_id: row.game_id,
    name: row.name,
    description: row.description || null,
    game_rules: row.game_rules || null,
    game_owner_name: row.game_owner_name || null,
    game_drive_link: row.game_drive_link || null,
    game_video_link: row.game_video_link || null,
    version: row.version,
    game_code: row.game_code,
    is_active: row.is_active,
    default_fps: row.default_fps ?? 5,
    baseline_actions: row.baseline_actions ?? null,
    tags: row.tags ?? [],
    grid_max_size: row.grid_max_size ?? 64,
    game_file_path: row.game_file_path,
    metadata_file_path: row.metadata_file_path,
    local_dir: row.local_dir,
    total_plays: row.total_plays ?? 0,
    total_wins: row.total_wins ?? 0,
    avg_score: row.avg_score ?? 0,
    created_at: row.created_at,
    updated_at: row.updated_at,
    uploaded_by: row.uploaded_by || null,
  };
}

function formatSession(row: any): any {
  return {
    player_name: row.player_name || "Anonymous",
    state: row.state,
    score: row.score,
    total_actions: row.total_actions,
    total_time: row.total_time,
    level: row.current_level,
    level_stats: row.level_stats ?? [],
    ended_at: row.ended_at || null,
  };
}

// ════════════════════════════════════════════════════════════════════════════
// PUBLIC ENDPOINTS (no auth required) — 7 endpoints
// ════════════════════════════════════════════════════════════════════════════

// ──── 1. GET /public — List active games ────
router.get(
  "/public",
  asyncHandler(async (_req, res) => {
    const rows = await queryAll(
      "SELECT * FROM games WHERE is_active = true ORDER BY created_at DESC",
    );

    res.json(rows.map(formatGame));
  }),
);

// ──── 2. GET /public/stats — Homepage stats ────
router.get(
  "/public/stats",
  asyncHandler(async (_req, res) => {
    const activeGames = await queryAll(
      "SELECT * FROM games WHERE is_active = true",
    );

    let totalPlays = 0;
    let totalWins = 0;
    for (const g of activeGames) {
      totalPlays += g.total_plays || 0;
      totalWins += g.total_wins || 0;
    }
    const winRate = totalPlays > 0 ? Math.round((totalWins / totalPlays * 100) * 10) / 10 : 0;

    // Sort by total_plays desc, take top 3
    const topPlayed = [...activeGames]
      .sort((a, b) => (b.total_plays || 0) - (a.total_plays || 0))
      .slice(0, 3);

    // Find top performer (fastest win) for each active game
    const topPerformers: Record<string, any> = {};
    for (const g of activeGames) {
      const fastest = await queryOne(
        `SELECT ps.*, u.username
         FROM play_sessions ps
         LEFT JOIN users u ON u.id = ps.user_id
         WHERE ps.game_id = $1 AND ps.state = 'WIN' AND ps.total_time > 0
         ORDER BY ps.total_time ASC
         LIMIT 1`,
        [g.id],
      );

      if (fastest) {
        topPerformers[g.game_id] = {
          player: fastest.player_name || fastest.username || "Anonymous",
          time: Math.round((fastest.total_time || 0) * 100) / 100,
          actions: fastest.total_actions || 0,
        };
      }
    }

    // Build top game stats
    const gameStats = topPlayed.map((g) => {
      const plays = g.total_plays || 0;
      const wins = g.total_wins || 0;
      const ba = g.baseline_actions;
      return {
        game_id: g.game_id,
        name: g.name || g.game_id,
        total_plays: plays,
        total_wins: wins,
        win_rate: plays > 0 ? Math.round((wins / plays * 100) * 10) / 10 : 0,
        top_performer: topPerformers[g.game_id] || null,
        levels: Array.isArray(ba) ? ba.length : 0,
      };
    });

    const featured = gameStats.length > 0 ? gameStats[0] : null;

    res.json({
      total_plays: totalPlays,
      total_wins: totalWins,
      win_rate: winRate,
      total_games: activeGames.length,
      featured_game: featured,
      top_games: gameStats,
    });
  }),
);

// ──── 3. GET /public/:gameId — Get single active game ────
router.get(
  "/public/:gameId",
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const row = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [gameId],
    );

    if (!row) {
      throw new AppError("Game not found", 404);
    }

    res.json(formatGame(row));
  }),
);

// ──── 4. GET /public/:gameId/plays — Recent plays / leaderboard ────
router.get(
  "/public/:gameId/plays",
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 20, 1), 100);

    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [gameId],
    );

    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const sessions = await queryAll(
      `SELECT * FROM play_sessions
       WHERE game_id = $1 AND state IN ('WIN', 'GAME_OVER')
       ORDER BY ended_at DESC
       LIMIT $2`,
      [game.id, limit],
    );

    res.json(sessions.map(formatSession));
  }),
);

// ──── 5. GET /public/:gameId/stats — Per-game stats ────
router.get(
  "/public/:gameId/stats",
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [gameId],
    );

    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const plays = game.total_plays || 0;
    const wins = game.total_wins || 0;
    const winRate = plays > 0 ? Math.round((wins / plays * 100) * 10) / 10 : 0;

    // Fastest win
    const fastest = await queryOne(
      `SELECT ps.*, u.username
       FROM play_sessions ps
       LEFT JOIN users u ON u.id = ps.user_id
       WHERE ps.game_id = $1 AND ps.state = 'WIN' AND ps.total_time > 0
       ORDER BY ps.total_time ASC
       LIMIT 1`,
      [game.id],
    );

    let topPerformer = null;
    if (fastest) {
      topPerformer = {
        player: fastest.player_name || fastest.username || "Anonymous",
        time: Math.round((fastest.total_time || 0) * 100) / 100,
        actions: fastest.total_actions || 0,
      };
    }

    // Average completion time (wins only)
    const avgRow = await queryOne(
      `SELECT AVG(total_time) as avg_time
       FROM play_sessions
       WHERE game_id = $1 AND state = 'WIN' AND total_time > 0`,
      [game.id],
    );
    const avgTime = avgRow?.avg_time ? Math.round(avgRow.avg_time * 100) / 100 : 0;

    // Recent players
    const recentRows = await queryAll(
      `SELECT ps.*, u.username
       FROM play_sessions ps
       LEFT JOIN users u ON u.id = ps.user_id
       WHERE ps.game_id = $1
       ORDER BY ps.started_at DESC
       LIMIT 5`,
      [game.id],
    );

    const recentPlayers = recentRows.map((s) => ({
      player: s.player_name || s.username || "Anonymous",
      state: s.state,
      time: Math.round((s.total_time || 0) * 100) / 100,
    }));

    const ba = game.baseline_actions;

    res.json({
      game_id: gameId,
      name: game.name,
      total_plays: plays,
      total_wins: wins,
      win_rate: winRate,
      avg_completion_time: avgTime,
      top_performer: topPerformer,
      recent_players: recentPlayers,
      levels: Array.isArray(ba) ? ba.length : 0,
    });
  }),
);

// ──── 6. GET /public/:gameId/preview — Initial grid frame for thumbnails ────
router.get(
  "/public/:gameId/preview",
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [gameId],
    );
    if (!game) throw new AppError("Game not found", 404);

    // Use Python bridge to get initial frame
    try {
      const { GamePythonBridge } = await import("../services/GamePythonBridge.js");
      const bridge = new GamePythonBridge();
      const gamePyPath = path.join(game.local_dir, `${game.game_code}.py`);
      const frame = await bridge.init(game.game_id, gamePyPath, 0);
      bridge.kill();
      
      res.json({
        game_id: gameId,
        grid: frame.grid,
        width: frame.width,
        height: frame.height,
      });
    } catch (err) {
      // Fallback: return empty grid
      res.json({
        game_id: gameId,
        grid: Array(8).fill(null).map(() => Array(8).fill(5)),
        width: 8,
        height: 8,
      });
    }
  }),
);

// ──── 7. Video Recording — Streaming chunks to server ────

// 7a. POST /public/:gameId/video/start — Begin a streaming recording session
router.post(
  "/public/:gameId/video/start",
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [gameId],
    );
    if (!game) throw new AppError("Game not found or inactive", 404);

    const playerName = (req.body.player_name || "").trim();
    const safeName = (playerName || "anonymous").replace(/[^a-zA-Z0-9_-]/g, "").substring(0, 20);
    const timestamp = new Date().toISOString().replace(/[-:T]/g, "").substring(0, 15).replace(/(\d{8})(\d{6})/, "$1_$2");
    const ext = req.body.ext || "webm";
    const filename = `${safeName}_${timestamp}.${ext}`;

    const recordingsDir = path.join(game.local_dir, "recordings");
    fs.mkdirSync(recordingsDir, { recursive: true });

    const filepath = path.join(recordingsDir, filename);
    // Create empty file
    fs.writeFileSync(filepath, Buffer.alloc(0));

    res.json({ recording_id: filename, filepath: filename, game_id: gameId });
  }),
);

// 7b. POST /public/:gameId/video/chunk — Append a chunk to a recording
router.post(
  "/public/:gameId/video/chunk",
  upload.single("chunk"),
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const recordingId = req.body.recording_id;
    if (!recordingId || !req.file) throw new AppError("recording_id and chunk required", 400);

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) throw new AppError("Game not found", 404);

    const safeFilename = path.basename(recordingId);
    const filepath = path.join(game.local_dir, "recordings", safeFilename);
    if (!fs.existsSync(filepath)) throw new AppError("Recording not found", 404);

    // Append chunk to file
    fs.appendFileSync(filepath, req.file.buffer);

    res.json({ ok: true, size: fs.statSync(filepath).size });
  }),
);

// 7c. POST /public/:gameId/video/end — Finalize a recording
router.post(
  "/public/:gameId/video/end",
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const recordingId = req.body.recording_id;
    if (!recordingId) throw new AppError("recording_id required", 400);

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) throw new AppError("Game not found", 404);

    const safeFilename = path.basename(recordingId);
    const filepath = path.join(game.local_dir, "recordings", safeFilename);

    if (!fs.existsSync(filepath)) throw new AppError("Recording not found", 404);

    const stat = fs.statSync(filepath);
    // Delete if empty (user cancelled without recording anything)
    if (stat.size === 0) {
      fs.unlinkSync(filepath);
      return res.json({ ok: true, deleted: true });
    }

    res.json({
      ok: true,
      filename: safeFilename,
      size: stat.size,
      game_id: gameId,
    });
  }),
);

// 7d. POST /public/:gameId/video — Legacy single-file upload (still supported for download-then-upload flow)
router.post(
  "/public/:gameId/video",
  upload.single("video"),
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 AND is_active = true",
      [gameId],
    );
    if (!game) throw new AppError("Game not found or inactive", 404);
    if (!req.file) throw new AppError("video file is required", 400);

    const recordingsDir = path.join(game.local_dir, "recordings");
    fs.mkdirSync(recordingsDir, { recursive: true });

    const playerName = (req.body.player_name || "").trim();
    const timestamp = new Date().toISOString().replace(/[-:T]/g, "").substring(0, 15).replace(/(\d{8})(\d{6})/, "$1_$2");
    const safeName = (playerName || "anonymous").replace(/[^a-zA-Z0-9_-]/g, "").substring(0, 20);
    const ext = req.body.ext || "webm";
    const filename = `${safeName}_${timestamp}.${ext}`;
    const filepath = path.join(recordingsDir, filename);

    fs.writeFileSync(filepath, req.file.buffer);

    res.json({ filename, size: fs.statSync(filepath).size, game_id: gameId });
  }),
);

// ════════════════════════════════════════════════════════════════════════════
// ADMIN ENDPOINTS (auth required) — 13 endpoints
// ════════════════════════════════════════════════════════════════════════════

// ──── 8. GET / — List all games ────
router.get(
  "/",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const activeOnly = req.query.active_only === "true";

    let rows: any[];
    if (activeOnly) {
      rows = await queryAll(
        "SELECT * FROM games WHERE is_active = true ORDER BY created_at DESC",
      );
    } else {
      rows = await queryAll(
        "SELECT * FROM games ORDER BY created_at DESC",
      );
    }

    res.json(rows.map(formatGame));
  }),
);

// ──── 9. GET /:gameId — Get game details ────
router.get(
  "/:gameId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const row = await queryOne(
      "SELECT * FROM games WHERE game_id = $1",
      [gameId],
    );

    if (!row) {
      throw new AppError("Game not found", 404);
    }

    res.json(formatGame(row));
  }),
);

// ──── 10. POST /upload — Upload new game (multipart) ────
router.post(
  "/upload",
  authenticateToken,
  requireAdmin,
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

    // Validate and write files to disk
    let result: any;
    try {
      result = uploadGameFiles(gamePyBytes, metadataBytes);
    } catch (e: any) {
      if (e instanceof AppError) throw e;
      throw new AppError(`Upload failed: ${e.message}`, 500);
    }

    // Check if game already exists
    const existing = await queryOne(
      "SELECT id FROM games WHERE game_id = $1",
      [result.game_id],
    );

    if (existing) {
      throw new AppError(
        `Game '${result.game_id}' already exists. Delete it first or use a different version.`,
        400,
      );
    }

    // Form fields
    const name = (req.body.name || "").trim() || result.game_id;
    const description = (req.body.description || "").trim() || null;
    const gameRules = (req.body.game_rules || "").trim() || null;
    const gameOwnerName = (req.body.game_owner_name || "").trim() || null;
    const gameDriveLink = (req.body.game_drive_link || "").trim() || null;
    const gameVideoLink = (req.body.game_video_link || "").trim() || null;

    const metadata = result.metadata;
    const id = genId();

    await pool.query(
      `INSERT INTO games (
        id, game_id, name, description, game_rules,
        game_owner_name, game_drive_link, game_video_link,
        version, game_code, is_active, default_fps,
        baseline_actions, tags,
        game_file_path, metadata_file_path, local_dir,
        uploaded_by, total_plays, total_wins
      ) VALUES (
        $1, $2, $3, $4, $5,
        $6, $7, $8,
        $9, $10, false, $11,
        $12, $13,
        $14, $15, $16,
        $17, 0, 0
      )`,
      [
        id,
        result.game_id,
        name,
        description,
        gameRules,
        gameOwnerName,
        gameDriveLink,
        gameVideoLink,
        result.version,
        result.game_code,
        metadata.default_fps ?? 5,
        toJsonb(metadata.baseline_actions ?? null),
        toJsonb(metadata.tags ?? null),
        result.game_file_path,
        result.metadata_file_path,
        result.local_dir,
        req.user.id,
      ],
    );

    const created = await queryOne("SELECT * FROM games WHERE id = $1", [id]);
    res.json(formatGame(created));
  }),
);

// ──── 11. PUT /:gameId — Update metadata ────
router.put(
  "/:gameId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const setClauses: string[] = [];
    const values: any[] = [];
    let paramIdx = 1;

    const fields: Record<string, string> = {
      name: "name",
      description: "description",
      game_rules: "game_rules",
      game_owner_name: "game_owner_name",
      game_drive_link: "game_drive_link",
      game_video_link: "game_video_link",
      default_fps: "default_fps",
    };

    for (const [bodyKey, dbCol] of Object.entries(fields)) {
      if (req.body[bodyKey] !== undefined) {
        setClauses.push(`${dbCol} = $${paramIdx++}`);
        values.push(req.body[bodyKey]);
      }
    }

    if (req.body.is_active !== undefined) {
      setClauses.push(`is_active = $${paramIdx++}`);
      values.push(!!req.body.is_active);
    }

    if (req.body.tags !== undefined) {
      setClauses.push(`tags = $${paramIdx++}`);
      values.push(toJsonb(req.body.tags ?? null));
    }

    if (setClauses.length === 0) {
      res.json(formatGame(game));
      return;
    }

    setClauses.push(`updated_at = NOW()`);
    values.push(game.id);

    await pool.query(
      `UPDATE games SET ${setClauses.join(", ")} WHERE id = $${paramIdx}`,
      values,
    );

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGame(updated));
  }),
);

// ──── 12. PUT /:gameId/files — Replace game files ────
router.put(
  "/:gameId/files",
  authenticateToken,
  requireAdmin,
  upload.fields([
    { name: "game_file", maxCount: 1 },
    { name: "metadata_file", maxCount: 1 },
  ]),
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const gameFile = (req.files as any)?.game_file?.[0];
    const metadataFile = (req.files as any)?.metadata_file?.[0];

    if (!gameFile && !metadataFile) {
      throw new AppError("No files provided", 400);
    }

    const gameDir = game.local_dir;
    if (!gameDir || !fs.existsSync(gameDir)) {
      throw new AppError("Game directory not found on disk", 500);
    }

    const setClauses: string[] = [];
    const values: any[] = [];
    let paramIdx = 1;

    // Update game.py if provided
    if (gameFile) {
      const gamePyPath = path.join(gameDir, `${game.game_code}.py`);
      fs.writeFileSync(gamePyPath, gameFile.buffer);
      setClauses.push(`game_file_path = $${paramIdx++}`);
      values.push(gamePyPath);
    }

    // Update metadata.json if provided
    if (metadataFile) {
      const metadataStr = metadataFile.buffer.toString("utf-8");
      let metadata: any;
      try {
        metadata = validateMetadata(metadataStr);
      } catch (e: any) {
        if (e instanceof AppError) throw e;
        throw new AppError(`Invalid metadata file: ${e.message}`, 400);
      }

      const metadataPath = path.join(gameDir, "metadata.json");
      fs.writeFileSync(metadataPath, metadataFile.buffer);
      setClauses.push(`metadata_file_path = $${paramIdx++}`);
      values.push(metadataPath);

      // Update game fields from new metadata
      if (metadata.default_fps) {
        setClauses.push(`default_fps = $${paramIdx++}`);
        values.push(metadata.default_fps);
      }
      if (metadata.baseline_actions !== undefined) {
        setClauses.push(`baseline_actions = $${paramIdx++}`);
        values.push(toJsonb(metadata.baseline_actions ?? null));
      }
      if (metadata.tags !== undefined) {
        setClauses.push(`tags = $${paramIdx++}`);
        values.push(toJsonb(metadata.tags ?? null));
      }
    }

    if (setClauses.length > 0) {
      setClauses.push(`updated_at = NOW()`);
      values.push(game.id);
      await pool.query(
        `UPDATE games SET ${setClauses.join(", ")} WHERE id = $${paramIdx}`,
        values,
      );
    }

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGame(updated));
  }),
);

// ──── 13. PATCH /:gameId/toggle — Activate/deactivate ────
router.patch(
  "/:gameId/toggle",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const isActive = req.body.is_active;
    if (isActive === undefined) {
      throw new AppError("is_active is required", 400);
    }

    await pool.query(
      "UPDATE games SET is_active = $1, updated_at = NOW() WHERE id = $2",
      [!!isActive, game.id],
    );

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGame(updated));
  }),
);

// ──── 14. DELETE /:gameId — Delete game + sessions ────
router.delete(
  "/:gameId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    // Delete files from disk
    deleteGameFiles(game.game_code, game.version);

    // Delete sessions and analytics
    await pool.query("DELETE FROM play_sessions WHERE game_id = $1", [game.id]);
    await pool.query("DELETE FROM game_analytics WHERE game_id = $1", [game.id]);

    // Delete the game record
    await pool.query("DELETE FROM games WHERE id = $1", [game.id]);

    res.json({ detail: `Game '${gameId}' deleted` });
  }),
);

// ──── 15. DELETE /:gameId/sessions — Clear sessions only ────
router.delete(
  "/:gameId/sessions",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const sessionCountRow = await queryOne(
      "SELECT COUNT(*) as cnt FROM play_sessions WHERE game_id = $1",
      [game.id],
    );
    const analyticsCountRow = await queryOne(
      "SELECT COUNT(*) as cnt FROM game_analytics WHERE game_id = $1",
      [game.id],
    );

    const sessionCount = sessionCountRow?.cnt ?? 0;
    const analyticsCount = analyticsCountRow?.cnt ?? 0;

    await pool.query("DELETE FROM play_sessions WHERE game_id = $1", [game.id]);
    await pool.query("DELETE FROM game_analytics WHERE game_id = $1", [game.id]);

    // Reset game counters
    await pool.query(
      "UPDATE games SET total_plays = 0, total_wins = 0, avg_score = 0, updated_at = NOW() WHERE id = $1",
      [game.id],
    );

    res.json({
      detail: `Cleared ${sessionCount} sessions and ${analyticsCount} analytics records`,
    });
  }),
);

// ──── 16. GET /:gameId/source — View source code ────
router.get(
  "/:gameId/source",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const sourceCode = getGameFileContent(game.game_code, game.version);
    const metadata = getMetadataContent(game.game_code, game.version);

    res.json({
      game_id: game.game_id,
      source_code: sourceCode,
      metadata,
    });
  }),
);

// ──── 17. POST /sync-local — Scan filesystem ────
router.post(
  "/sync-local",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const localGames = listLocalGames();
    const added: string[] = [];

    for (const lg of localGames) {
      const existing = await queryOne(
        "SELECT id FROM games WHERE game_id = $1",
        [lg.game_id],
      );

      if (!existing) {
        const id = genId();
        const gamePyPath = path.join(lg.local_dir, `${lg.game_code}.py`);
        const metadataPath = path.join(lg.local_dir, "metadata.json");

        await pool.query(
          `INSERT INTO games (
            id, game_id, name, version, game_code,
            game_file_path, metadata_file_path, local_dir,
            default_fps, baseline_actions, tags,
            uploaded_by, total_plays, total_wins
          ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8,
            $9, $10, $11,
            $12, 0, 0
          )`,
          [
            id,
            lg.game_id,
            lg.game_id, // name defaults to game_id
            lg.version,
            lg.game_code,
            gamePyPath,
            metadataPath,
            lg.local_dir,
            lg.metadata.default_fps ?? 5,
            toJsonb(lg.metadata.baseline_actions ?? null),
            toJsonb(lg.metadata.tags ?? null),
            req.user.id,
          ],
        );

        added.push(lg.game_id);
      }
    }

    res.json({ synced: added.length, games: added });
  }),
);

// ──── 18. GET /:gameId/videos — List recordings (admin) ────
router.get(
  "/:gameId/videos",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    const recordingsDir = path.join(game.local_dir, "recordings");
    if (!fs.existsSync(recordingsDir)) {
      res.json([]);
      return;
    }

    const entries = fs.readdirSync(recordingsDir).sort().reverse();
    const videos: any[] = [];

    for (const filename of entries) {
      if (!filename.endsWith(".webm")) continue;

      const filepath = path.join(recordingsDir, filename);
      const stat = fs.statSync(filepath);

      // Parse player name from filename: player_YYYYMMDD_HHMMSS.webm
      const parts = filename.replace(/\.webm$/, "").split("_");
      const player = parts.length >= 3 ? parts.slice(0, -2).join("_") : "unknown";

      videos.push({
        filename,
        player,
        size: stat.size,
        created_at: stat.mtime.toISOString(),
        url: `/api/games/${gameId}/videos/${filename}`,
      });
    }

    res.json(videos);
  }),
);

// ──── 19. GET /:gameId/videos/:filename — Serve video (admin) ────
router.get(
  "/:gameId/videos/:filename",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId, filename } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    // Sanitize filename to prevent path traversal
    const safeFilename = path.basename(filename);
    const filepath = path.join(game.local_dir, "recordings", safeFilename);

    if (!fs.existsSync(filepath)) {
      throw new AppError("Video not found", 404);
    }

    res.setHeader("Content-Type", "video/webm");
    res.setHeader("Content-Disposition", `inline; filename="${safeFilename}"`);
    const stream = fs.createReadStream(filepath);
    stream.pipe(res);
  }),
);

// ──── 20. DELETE /:gameId/videos/:filename — Delete video (admin) ────
router.delete(
  "/:gameId/videos/:filename",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId, filename } = req.params;

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1", [gameId]);
    if (!game) {
      throw new AppError("Game not found", 404);
    }

    // Sanitize filename to prevent path traversal
    const safeFilename = path.basename(filename);
    const filepath = path.join(game.local_dir, "recordings", safeFilename);

    if (!fs.existsSync(filepath)) {
      throw new AppError("Video not found", 404);
    }

    fs.unlinkSync(filepath);
    res.json({ detail: `Video '${safeFilename}' deleted` });
  }),
);

export default router;
