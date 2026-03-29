import { Router } from "express";
import fs from "fs";
import path from "path";
import multer from "multer";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireAdmin } from "../middleware/auth.js";
import pool, { genId, queryOne, queryAll, toJsonb } from "../db.js";
import { AppError } from "../middleware/errorHandler.js";

const router = Router();

const ENVIRONMENT_FILES_DIR = process.env.ENVIRONMENT_FILES_DIR ||
  path.join(process.cwd(), "environment_files");

const REQUESTS_DIR = path.join(ENVIRONMENT_FILES_DIR, "_requests");

const upload = multer({ storage: multer.memoryStorage() });

/** Format a game_request row from PostgreSQL for JSON response. */
function formatRequest(row: any) {
  return {
    id: row.id,
    game_id: row.game_id,
    requester_name: row.requester_name,
    requester_email: row.requester_email || null,
    message: row.message || null,
    description: row.description || null,
    game_rules: row.game_rules || null,
    game_owner_name: row.game_owner_name || null,
    game_drive_link: row.game_drive_link || null,
    game_video_link: row.game_video_link || null,
    status: row.status,
    admin_note: row.admin_note || null,
    game_code: row.game_code || null,
    version: row.version,
    tags: row.tags ?? null,
    default_fps: row.default_fps ?? 5,
    created_at: row.created_at,
    reviewed_at: row.reviewed_at || null,
  };
}

/** Format a game row from PostgreSQL for JSON response. */
function formatGame(row: any) {
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
    tags: row.tags ?? null,
    grid_max_size: row.grid_max_size ?? 64,
    total_plays: row.total_plays ?? 0,
    total_wins: row.total_wins ?? 0,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

// ──── URL validation (SSRF protection) ────

async function validateUrl(url: string, fieldName: string): Promise<void> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new AppError(`${fieldName}: URL must start with http:// or https://`, 400);
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new AppError(`${fieldName}: URL must start with http:// or https://`, 400);
  }

  const hostname = parsed.hostname;
  if (!hostname) {
    throw new AppError(`${fieldName}: URL has no valid hostname`, 400);
  }

  if (hostname === "localhost" || hostname === "localhost.localdomain") {
    throw new AppError(`${fieldName}: URLs targeting localhost are not allowed`, 400);
  }

  // DNS resolution + private IP check
  const dns = await import("dns");
  try {
    const { resolve4 } = dns.promises;
    const addresses = await resolve4(hostname);
    for (const addr of addresses) {
      if (isPrivateIp(addr)) {
        throw new AppError(
          `${fieldName}: URLs targeting private/internal networks are not allowed`,
          400,
        );
      }
    }
  } catch (err: any) {
    // If it's our own AppError, rethrow it
    if (err instanceof AppError) throw err;
    // DNS resolution failed — let it pass (don't block on transient DNS issues)
  }

  // HEAD request to check for 404
  try {
    const response = await fetch(url, {
      method: "HEAD",
      signal: AbortSignal.timeout(10000),
      redirect: "follow",
    });
    if (response.status === 404) {
      throw new AppError(
        `${fieldName}: URL returned 404 (not found). Please check the link.`,
        400,
      );
    }
  } catch (err: any) {
    // If it's our own AppError, rethrow it
    if (err instanceof AppError) throw err;
    // Network error / timeout — let it pass, don't block upload
  }
}

function isPrivateIp(ip: string): boolean {
  const parts = ip.split(".").map(Number);
  if (parts.length !== 4) return false;
  // 10.x.x.x
  if (parts[0] === 10) return true;
  // 172.16.x.x - 172.31.x.x
  if (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) return true;
  // 192.168.x.x
  if (parts[0] === 192 && parts[1] === 168) return true;
  // 127.x.x.x (loopback)
  if (parts[0] === 127) return true;
  // 169.254.x.x (link-local)
  if (parts[0] === 169 && parts[1] === 254) return true;
  // 0.x.x.x
  if (parts[0] === 0) return true;
  return false;
}

// ──── POST /submit — Public: submit a game for review (multipart file upload) ────
router.post(
  "/submit",
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

    // Form fields
    const requesterName = (req.body.requester_name || "").trim();
    const requesterEmail = (req.body.requester_email || "").trim() || null;
    const message = (req.body.message || "").trim() || null;
    const description = (req.body.description || "").trim() || null;
    const gameRules = (req.body.game_rules || "").trim() || null;
    const gameOwnerName = (req.body.game_owner_name || "").trim() || null;
    const gameDriveLink = (req.body.game_drive_link || "").trim() || null;
    const gameVideoLink = (req.body.game_video_link || "").trim() || null;

    if (!requesterName) {
      throw new AppError("requester_name is required", 400);
    }

    // Validate URLs before processing files
    if (gameDriveLink) {
      await validateUrl(gameDriveLink, "Game drive link");
    }
    if (gameVideoLink) {
      await validateUrl(gameVideoLink, "Game video link");
    }

    const gamePyBytes: Buffer = gameFile.buffer;
    const metadataBytes: Buffer = metadataFile.buffer;

    let metadata: any;
    try {
      metadata = JSON.parse(metadataBytes.toString("utf-8"));
    } catch {
      throw new AppError("Invalid metadata.json", 400);
    }

    const gameId: string | undefined = metadata.game_id;
    if (!gameId) {
      throw new AppError('metadata.json must contain "game_id"', 400);
    }

    // Check uniqueness: game_id must not exist in games or pending requests
    const existingGame = await queryOne(
      "SELECT id FROM games WHERE game_id = $1",
      [gameId],
    );
    if (existingGame) {
      throw new AppError(`Game "${gameId}" already exists`, 400);
    }

    const existingRequest = await queryOne(
      "SELECT id FROM game_requests WHERE game_id = $1 AND status = 'pending'",
      [gameId],
    );
    if (existingRequest) {
      throw new AppError(
        `A request for "${gameId}" is already pending review`,
        400,
      );
    }

    let gameCode: string;
    let version: string;
    if (gameId.includes("-")) {
      // rsplit equivalent: split from the right at last dash
      const lastDash = gameId.lastIndexOf("-");
      gameCode = gameId.substring(0, lastDash);
      version = gameId.substring(lastDash + 1) || "v1";
    } else {
      gameCode = gameId;
      version = "v1";
    }

    // Save files to _requests directory
    const reqDir = path.join(REQUESTS_DIR, gameId);
    fs.mkdirSync(reqDir, { recursive: true });

    const gamePyPath = path.join(reqDir, `${gameCode}.py`);
    const metadataPath = path.join(reqDir, "metadata.json");

    fs.writeFileSync(gamePyPath, gamePyBytes);
    fs.writeFileSync(metadataPath, metadataBytes);

    const id = genId();
    await pool.query(`
      INSERT INTO game_requests (
        id, game_id, requester_name, requester_email, message,
        description, game_rules, game_owner_name, game_drive_link, game_video_link,
        status, game_file_path, metadata_file_path, local_dir,
        game_file_content, metadata_file_content,
        game_code, version, tags, default_fps
      ) VALUES (
        $1, $2, $3, $4, $5,
        $6, $7, $8, $9, $10,
        'pending', $11, $12, $13,
        $14, $15,
        $16, $17, $18, $19
      )
    `, [
      id,
      gameId,
      requesterName,
      requesterEmail,
      message,
      description,
      gameRules,
      gameOwnerName,
      gameDriveLink,
      gameVideoLink,
      gamePyPath,
      metadataPath,
      reqDir,
      gamePyBytes,
      metadataBytes,
      gameCode,
      version,
      toJsonb(metadata.tags ?? null),
      metadata.default_fps ?? 5,
    ]);

    const created = await queryOne(
      "SELECT * FROM game_requests WHERE id = $1",
      [id],
    );

    res.json(formatRequest(created));
  }),
);

// ──── GET / — Admin: list requests by status ────
router.get(
  "/",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const status = (req.query.status as string) || "pending";

    let rows: any[];
    if (status === "all") {
      rows = await queryAll(
        "SELECT * FROM game_requests ORDER BY created_at DESC",
      );
    } else {
      rows = await queryAll(
        "SELECT * FROM game_requests WHERE status = $1 ORDER BY created_at DESC",
        [status],
      );
    }

    res.json(rows.map(formatRequest));
  }),
);

// ──── GET /:requestId — Admin: get request details ────
router.get(
  "/:requestId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { requestId } = req.params;
    const row = await queryOne(
      "SELECT * FROM game_requests WHERE id = $1",
      [requestId],
    );

    if (!row) {
      throw new AppError("Request not found", 404);
    }

    res.json(formatRequest(row));
  }),
);

// ──── GET /:requestId/source — Admin: view submitted source code ────
router.get(
  "/:requestId/source",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { requestId } = req.params;
    const row = await queryOne(
      "SELECT * FROM game_requests WHERE id = $1",
      [requestId],
    );

    if (!row) {
      throw new AppError("Request not found", 404);
    }

    let sourceCode: string | null = null;
    let metadata: any = null;

    // Try blob content first, then fall back to file path
    if (row.game_file_content) {
      sourceCode = Buffer.isBuffer(row.game_file_content)
        ? row.game_file_content.toString("utf-8")
        : String(row.game_file_content);
    } else if (row.game_file_path && fs.existsSync(row.game_file_path)) {
      sourceCode = fs.readFileSync(row.game_file_path, "utf-8");
    }

    if (row.metadata_file_content) {
      const raw = Buffer.isBuffer(row.metadata_file_content)
        ? row.metadata_file_content.toString("utf-8")
        : String(row.metadata_file_content);
      try {
        metadata = JSON.parse(raw);
      } catch {
        metadata = null;
      }
    } else if (
      row.metadata_file_path &&
      fs.existsSync(row.metadata_file_path)
    ) {
      try {
        metadata = JSON.parse(
          fs.readFileSync(row.metadata_file_path, "utf-8"),
        );
      } catch {
        metadata = null;
      }
    }

    res.json({ source_code: sourceCode, metadata });
  }),
);

// ──── GET /:requestId/files/:fileType — Admin: download submitted file ────
router.get(
  "/:requestId/files/:fileType",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { requestId, fileType } = req.params;
    const row = await queryOne(
      "SELECT * FROM game_requests WHERE id = $1",
      [requestId],
    );

    if (!row) {
      throw new AppError("Request not found", 404);
    }

    if (fileType === "game") {
      let content: Buffer | null = null;
      if (row.game_file_content) {
        content = Buffer.isBuffer(row.game_file_content)
          ? row.game_file_content
          : Buffer.from(row.game_file_content);
      } else if (
        row.game_file_path &&
        fs.existsSync(row.game_file_path)
      ) {
        content = fs.readFileSync(row.game_file_path);
      }

      if (!content) {
        throw new AppError("Game file not found", 404);
      }

      res.setHeader("Content-Type", "text/x-python");
      res.setHeader(
        "Content-Disposition",
        `attachment; filename="${row.game_id}.py"`,
      );
      res.send(content);
    } else if (fileType === "metadata") {
      let content: Buffer | null = null;
      if (row.metadata_file_content) {
        content = Buffer.isBuffer(row.metadata_file_content)
          ? row.metadata_file_content
          : Buffer.from(row.metadata_file_content);
      } else if (
        row.metadata_file_path &&
        fs.existsSync(row.metadata_file_path)
      ) {
        content = fs.readFileSync(row.metadata_file_path);
      }

      if (!content) {
        throw new AppError("Metadata file not found", 404);
      }

      res.setHeader("Content-Type", "application/json");
      res.setHeader(
        "Content-Disposition",
        'attachment; filename="metadata.json"',
      );
      res.send(content);
    } else {
      throw new AppError("file_type must be 'game' or 'metadata'", 400);
    }
  }),
);

// ──── POST /:requestId/review — Admin: approve or reject ────
router.post(
  "/:requestId/review",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { requestId } = req.params;
    const { action, admin_note } = req.body;

    const row = await queryOne(
      "SELECT * FROM game_requests WHERE id = $1",
      [requestId],
    );

    if (!row) {
      throw new AppError("Request not found", 404);
    }

    if (row.status !== "pending") {
      throw new AppError(`Request already ${row.status}`, 400);
    }

    // ── Reject ──
    if (action === "reject") {
      await pool.query(
        "UPDATE game_requests SET status = 'rejected', admin_note = $1, reviewed_at = NOW(), reviewed_by = $2 WHERE id = $3",
        [admin_note || null, req.user.id, requestId],
      );

      const updated = await queryOne(
        "SELECT * FROM game_requests WHERE id = $1",
        [requestId],
      );
      res.json(formatRequest(updated));
      return;
    }

    if (action !== "approve") {
      throw new AppError('action must be "approve" or "reject"', 400);
    }

    // ── Approve ──

    // Check game doesn't already exist
    const existingGame = await queryOne(
      "SELECT id FROM games WHERE game_id = $1",
      [row.game_id],
    );
    if (existingGame) {
      throw new AppError(`Game "${row.game_id}" already exists`, 400);
    }

    // Get file content from blob or disk
    let gamePyBytes: Buffer | null = row.game_file_content
      ? Buffer.isBuffer(row.game_file_content)
        ? row.game_file_content
        : Buffer.from(row.game_file_content)
      : null;
    let metadataBytes: Buffer | null = row.metadata_file_content
      ? Buffer.isBuffer(row.metadata_file_content)
        ? row.metadata_file_content
        : Buffer.from(row.metadata_file_content)
      : null;

    if (!gamePyBytes || !metadataBytes) {
      try {
        if (!gamePyBytes && row.game_file_path) {
          gamePyBytes = fs.readFileSync(row.game_file_path);
        }
        if (!metadataBytes && row.metadata_file_path) {
          metadataBytes = fs.readFileSync(row.metadata_file_path);
        }
      } catch (e: any) {
        throw new AppError(
          `Game files not found. The request may need to be resubmitted: ${e.message}`,
          500,
        );
      }
    }

    if (!gamePyBytes || !metadataBytes) {
      throw new AppError(
        "Game files not found. The request may need to be resubmitted.",
        500,
      );
    }

    // Parse metadata
    let metadata: any;
    try {
      metadata = JSON.parse(metadataBytes.toString("utf-8"));
    } catch {
      throw new AppError("Invalid metadata.json in request", 400);
    }

    const fullGameId = metadata.game_id || row.game_id;
    let gameCode: string;
    if (fullGameId.includes("-")) {
      const lastDash = fullGameId.lastIndexOf("-");
      gameCode = fullGameId.substring(0, lastDash);
    } else {
      gameCode = fullGameId;
    }

    const gameDir = path.join(ENVIRONMENT_FILES_DIR, gameCode);
    fs.mkdirSync(gameDir, { recursive: true });

    const gamePyPath = path.join(gameDir, `${gameCode}.py`);
    const metadataFilePath = path.join(gameDir, "metadata.json");

    const gamePyStr = gamePyBytes.toString("utf-8");
    fs.writeFileSync(gamePyPath, gamePyStr, "utf-8");

    metadata.local_dir = path.join("environment_files", gameCode);
    fs.writeFileSync(metadataFilePath, JSON.stringify(metadata, null, 2), "utf-8");

    // Create game record in DB
    const gameDbId = genId();
    await pool.query(`
      INSERT INTO games (
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
      )
    `, [
      gameDbId,
      fullGameId,
      fullGameId, // name = game_id
      row.description || null,
      row.game_rules || null,
      row.game_owner_name || null,
      row.game_drive_link || null,
      row.game_video_link || null,
      "v1",
      gameCode,
      metadata.default_fps ?? 5,
      toJsonb(metadata.baseline_actions ?? null),
      toJsonb(metadata.tags ?? null),
      gamePyPath,
      metadataFilePath,
      gameDir,
      req.user.id,
    ]);

    // Delete the request from DB
    await pool.query("DELETE FROM game_requests WHERE id = $1", [requestId]);

    // Clean up request files
    if (row.local_dir && fs.existsSync(row.local_dir)) {
      fs.rmSync(row.local_dir, { recursive: true, force: true });
    }

    // Return the created game
    const game = await queryOne(
      "SELECT * FROM games WHERE id = $1",
      [gameDbId],
    );

    res.json(formatGame(game));
  }),
);

// ──── DELETE /:requestId — Admin: delete request ────
router.delete(
  "/:requestId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { requestId } = req.params;
    const row = await queryOne(
      "SELECT * FROM game_requests WHERE id = $1",
      [requestId],
    );

    if (!row) {
      throw new AppError("Request not found", 404);
    }

    // Clean up request files
    if (row.local_dir && fs.existsSync(row.local_dir)) {
      fs.rmSync(row.local_dir, { recursive: true, force: true });
    }

    await pool.query("DELETE FROM game_requests WHERE id = $1", [requestId]);

    res.json({ detail: "Request deleted" });
  }),
);

export default router;
