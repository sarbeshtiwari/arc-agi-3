import { Router } from "express";
import ExcelJS from "exceljs";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireAdmin } from "../middleware/auth.js";
import { AppError } from "../middleware/errorHandler.js";
import pool, { queryOne, queryAll } from "../db.js";

const router = Router();

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
    id: row.id,
    game_name: row.game_name || row.name || "Unknown",
    game_id: row.game_game_id || row.game_id,
    player: row.player_name || "Anonymous",
    state: row.state,
    total_actions: row.total_actions,
    total_time: row.total_time || 0,
    level_stats: row.level_stats ?? [],
    current_level: row.current_level,
    game_overs: row.game_overs || 0,
    started_at: row.started_at,
  };
}

/**
 * Format level_stats JSON array into a human-readable string for Excel export.
 */
function formatLevelStatsStr(levelStats: any[]): string {
  if (!levelStats || !Array.isArray(levelStats) || levelStats.length === 0) return "";
  const parts = levelStats.map((ls: any) => {
    const status = ls.completed ? "Done" : "Incomplete";
    const level = (ls.level ?? 0) + 1;
    const actions = ls.actions ?? 0;
    const time = Math.round((ls.time ?? 0) * 10) / 10;
    const deaths = ls.game_overs ?? 0;
    const resets = ls.resets ?? 0;
    return `Lv${level}: ${actions}moves, ${time}s, ${deaths}deaths, ${resets}resets (${status})`;
  });
  return parts.join(" | ");
}

// ════════════════════════════════════════════════════════════════════════════
// AUTH REQUIRED ENDPOINTS (1-4)
// ════════════════════════════════════════════════════════════════════════════

// ──── 1. GET /dashboard — Dashboard overview statistics ────
router.get(
  "/dashboard",
  authenticateToken,
  asyncHandler(async (_req, res) => {
    // Counts
    const totalGames = (
      await queryOne("SELECT COUNT(*) as cnt FROM games", [])
    ).cnt;
    const activeGames = (
      await queryOne("SELECT COUNT(*) as cnt FROM games WHERE is_active = true", [])
    ).cnt;
    const totalPlays = (
      await queryOne("SELECT COUNT(*) as cnt FROM play_sessions", [])
    ).cnt;
    const totalUsers = (
      await queryOne("SELECT COUNT(*) as cnt FROM users", [])
    ).cnt;
    const totalWins = (
      await queryOne("SELECT COUNT(*) as cnt FROM play_sessions WHERE state = 'WIN'", [])
    ).cnt;
    const totalGameOvers = (
      await queryOne("SELECT COUNT(*) as cnt FROM play_sessions WHERE state = 'GAME_OVER'", [])
    ).cnt;
    const pendingRequests = (
      await queryOne("SELECT COUNT(*) as cnt FROM game_requests WHERE status = 'pending'", [])
    ).cnt;

    const winRate = totalPlays > 0 ? Math.round((totalWins / totalPlays) * 1000) / 10 : 0;

    // Daily plays/wins for last 7 days
    const dailyRows = await queryAll(
      `SELECT started_at::date as day,
              COUNT(*) as plays,
              SUM(CASE WHEN state = 'WIN' THEN 1 ELSE 0 END) as wins
       FROM play_sessions
       WHERE started_at >= NOW() - INTERVAL '7 days'
       GROUP BY day
       ORDER BY day`,
      [],
    );

    // Build arrays indexed by day for the last 7 days
    const dailyMap: Record<string, { plays: number; wins: number }> = {};
    for (const row of dailyRows) {
      // day comes back as a Date object from pg; convert to YYYY-MM-DD string
      const dayStr = row.day instanceof Date
        ? row.day.toISOString().slice(0, 10)
        : String(row.day).slice(0, 10);
      dailyMap[dayStr] = { plays: Number(row.plays), wins: Number(row.wins) };
    }

    const dailyPlays: number[] = [];
    const dailyWins: number[] = [];
    const dailyLabels: string[] = [];
    const now = new Date();
    for (let i = 6; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      const dayStr = d.toISOString().slice(0, 10); // YYYY-MM-DD
      const label = d.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
      const entry = dailyMap[dayStr] || { plays: 0, wins: 0 };
      dailyPlays.push(entry.plays);
      dailyWins.push(entry.wins);
      dailyLabels.push(label);
    }

    // Game distribution (top 8 games by plays)
    const distRows = await queryAll(
      `SELECT name, game_id, total_plays, total_wins
       FROM games
       WHERE total_plays > 0
       ORDER BY total_plays DESC
       LIMIT 8`,
      [],
    );

    const gameDistribution = distRows.map((g: any) => ({
      name: g.name || g.game_id,
      plays: g.total_plays || 0,
      wins: g.total_wins || 0,
    }));

    // Recent games (last 5 created)
    const recentGameRows = await queryAll(
      "SELECT * FROM games ORDER BY created_at DESC LIMIT 5",
      [],
    );

    // Top played games (top 5)
    const topPlayedRows = await queryAll(
      "SELECT * FROM games ORDER BY total_plays DESC LIMIT 5",
      [],
    );

    const topPlayedGames = topPlayedRows.map((g: any) => ({
      game_id: g.game_id,
      name: g.name,
      total_plays: g.total_plays ?? 0,
      total_wins: g.total_wins ?? 0,
    }));

    res.json({
      total_games: totalGames,
      active_games: activeGames,
      total_plays: totalPlays,
      total_users: totalUsers,
      total_wins: totalWins,
      total_game_overs: totalGameOvers,
      pending_requests: pendingRequests,
      win_rate: winRate,
      daily_plays: dailyPlays,
      daily_wins: dailyWins,
      daily_labels: dailyLabels,
      game_distribution: gameDistribution,
      recent_games: recentGameRows.map(formatGame),
      top_played_games: topPlayedGames,
    });
  }),
);

// ──── 2. GET /game/:gameId — Per-game analytics ────
router.get(
  "/game/:gameId",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const days = Math.min(Math.max(parseInt(req.query.days as string) || 30, 1), 365);

    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1",
      [gameId],
    );

    if (!game) {
      throw new AppError("Game not found", 404);
    }

    // Fetch sessions within the date range
    const sessions = await queryAll(
      `SELECT * FROM play_sessions
       WHERE game_id = $1 AND started_at >= NOW() - ($2 || ' days')::interval
       ORDER BY started_at`,
      [game.id, String(days)],
    );

    const total = sessions.length;
    const wins = sessions.filter((s: any) => s.state === "WIN").length;
    const actionsToWin = sessions
      .filter((s: any) => s.state === "WIN")
      .map((s: any) => s.total_actions);
    const timesToWin = sessions
      .filter((s: any) => s.state === "WIN" && s.total_time > 0)
      .map((s: any) => s.total_time);

    const uniquePlayers = new Set(
      sessions.map((s: any) => s.player_name || s.user_id || "anon"),
    ).size;

    // Build daily stats
    const dailyMap: Record<string, { date: string; plays: number; wins: number; times: number[] }> = {};
    for (const s of sessions) {
      const startedAt = s.started_at instanceof Date
        ? s.started_at.toISOString().slice(0, 10)
        : (s.started_at || "").slice(0, 10);
      if (!startedAt) continue;
      if (!dailyMap[startedAt]) {
        dailyMap[startedAt] = { date: startedAt, plays: 0, wins: 0, times: [] };
      }
      dailyMap[startedAt].plays += 1;
      if (s.state === "WIN") {
        dailyMap[startedAt].wins += 1;
      }
      if (s.total_time > 0) {
        dailyMap[startedAt].times.push(s.total_time);
      }
    }

    const dailyStats = Object.values(dailyMap)
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((d) => ({
        date: d.date,
        plays: d.plays,
        wins: d.wins,
        avg_time:
          d.times.length > 0
            ? Math.round((d.times.reduce((a, b) => a + b, 0) / d.times.length) * 10) / 10
            : 0,
      }));

    const avgActionsToWin =
      actionsToWin.length > 0
        ? Math.round((actionsToWin.reduce((a, b) => a + b, 0) / actionsToWin.length) * 10) / 10
        : 0;
    const avgTimeToWin =
      timesToWin.length > 0
        ? Math.round((timesToWin.reduce((a, b) => a + b, 0) / timesToWin.length) * 10) / 10
        : 0;

    res.json({
      game_id: game.game_id,
      game_name: game.name,
      total_plays: total,
      total_wins: wins,
      win_rate: total > 0 ? Math.round((wins / total) * 10000) / 100 : 0,
      avg_actions_to_win: avgActionsToWin,
      avg_time_to_win: avgTimeToWin,
      unique_players: uniquePlayers,
      daily_stats: dailyStats,
    });
  }),
);

// ──── 3. GET /sessions — Recent play sessions ────
router.get(
  "/sessions",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const gameId = req.query.game_id as string | undefined;
    const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 20, 1), 100);

    let gameDbId: string | null = null;
    if (gameId) {
      const game = await queryOne(
        "SELECT id FROM games WHERE game_id = $1",
        [gameId],
      );
      if (game) {
        gameDbId = game.id;
      }
    }

    let sessions: any[];
    if (gameDbId) {
      sessions = await queryAll(
        `SELECT ps.*, g.name as game_name, g.game_id as game_game_id, u.username as username
         FROM play_sessions ps
         LEFT JOIN games g ON g.id = ps.game_id
         LEFT JOIN users u ON u.id = ps.user_id
         WHERE ps.game_id = $1
         ORDER BY ps.started_at DESC
         LIMIT $2`,
        [gameDbId, limit],
      );
    } else {
      sessions = await queryAll(
        `SELECT ps.*, g.name as game_name, g.game_id as game_game_id, u.username as username
         FROM play_sessions ps
         LEFT JOIN games g ON g.id = ps.game_id
         LEFT JOIN users u ON u.id = ps.user_id
         ORDER BY ps.started_at DESC
         LIMIT $1`,
        [limit],
      );
    }

    const results = sessions.map((s: any) => ({
      id: s.id,
      game_name: s.game_name || "Unknown",
      game_id: s.game_game_id || s.game_id,
      player: s.player_name || s.username || "Anonymous",
      state: s.state,
      score: s.score || 0,
      total_actions: s.total_actions,
      total_time: s.total_time || 0,
      game_overs: s.game_overs || 0,
      level_stats: s.level_stats ?? [],
      current_level: s.current_level,
      started_at: s.started_at,
      ended_at: s.ended_at || null,
    }));

    res.json(results);
  }),
);

// ──── 4. GET /replay/:sessionId — Replay data for a session ────
router.get(
  "/replay/:sessionId",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const { sessionId } = req.params;

    const session = await queryOne(
      "SELECT * FROM play_sessions WHERE id = $1",
      [sessionId],
    );

    if (!session) {
      throw new AppError("Session not found", 404);
    }

    const game = await queryOne(
      "SELECT * FROM games WHERE id = $1",
      [session.game_id],
    );

    res.json({
      session_id: session.id,
      game_id: game ? game.game_id : "unknown",
      game_name: game ? game.name : "Unknown",
      seed: session.seed,
      state: session.state,
      total_actions: session.total_actions,
      total_time: session.total_time,
      level_stats: session.level_stats ?? [],
      player_name: session.player_name || "Anonymous",
      action_log: session.action_log ?? [],
      started_at: session.started_at,
      ended_at: session.ended_at || null,
    });
  }),
);

// ════════════════════════════════════════════════════════════════════════════
// ADMIN-ONLY ENDPOINTS (5-9)
// ════════════════════════════════════════════════════════════════════════════

// ──── 5. GET /temp-sessions — List ephemeral sessions ────
router.get(
  "/temp-sessions",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (_req, res) => {
    const sessions = await queryAll(
      "SELECT * FROM temp_game_sessions ORDER BY started_at DESC",
      [],
    );

    res.json(
      sessions.map((s: any) => ({
        id: s.id,
        session_guid: s.session_guid,
        game_id: s.game_id,
        player_name: s.player_name || "Anonymous",
        state: s.state,
        score: s.score || 0,
        total_actions: s.total_actions || 0,
        current_level: s.current_level || 0,
        game_overs: s.game_overs || 0,
        total_time: s.total_time || 0,
        level_stats: s.level_stats ?? [],
        action_log: s.action_log ?? [],
        started_at: s.started_at || null,
        ended_at: s.ended_at || null,
        expires_at: s.expires_at || null,
      })),
    );
  }),
);

// ──── 6. DELETE /temp-sessions — Delete all temp sessions ────
router.delete(
  "/temp-sessions",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (_req, res) => {
    const countRow = await queryOne(
      "SELECT COUNT(*) as cnt FROM temp_game_sessions",
      [],
    );
    const count = countRow.cnt;

    await pool.query("DELETE FROM temp_game_sessions");

    res.json({ detail: `Deleted ${count} temp session(s)` });
  }),
);

// ──── 7. GET /export/:gameId — Export sessions as Excel ────
router.get(
  "/export/:gameId",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const { gameId } = req.params;
    const filter = (req.query.filter as string) || "all";
    const date = req.query.date as string | undefined;
    const dateFrom = req.query.date_from as string | undefined;
    const dateTo = req.query.date_to as string | undefined;

    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1",
      [gameId],
    );

    if (!game) {
      throw new AppError("Game not found", 404);
    }

    // Build query with date filters
    let sql = "SELECT * FROM play_sessions WHERE game_id = $1";
    const params: any[] = [game.id];
    let paramIdx = 2;

    if (filter === "today") {
      sql += " AND started_at >= CURRENT_DATE";
    } else if (filter === "date" && date) {
      // Validate date format
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
        throw new AppError("Invalid date format. Use YYYY-MM-DD", 400);
      }
      sql += ` AND started_at >= $${paramIdx} AND started_at < ($${paramIdx + 1}::date + INTERVAL '1 day')`;
      params.push(date, date);
      paramIdx += 2;
    } else if (filter === "range" && dateFrom && dateTo) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(dateFrom) || !/^\d{4}-\d{2}-\d{2}$/.test(dateTo)) {
        throw new AppError("Invalid date format. Use YYYY-MM-DD", 400);
      }
      sql += ` AND started_at >= $${paramIdx} AND started_at < ($${paramIdx + 1}::date + INTERVAL '1 day')`;
      params.push(dateFrom, dateTo);
      paramIdx += 2;
    }

    sql += " ORDER BY started_at DESC";

    const sessions = (await pool.query(sql, params)).rows;

    // Pre-fetch users for player names
    const userIds = [...new Set(sessions.filter((s: any) => s.user_id).map((s: any) => s.user_id))];
    const usersMap: Record<string, any> = {};
    for (const uid of userIds) {
      const u = await queryOne("SELECT * FROM users WHERE id = $1", [uid]);
      if (u) usersMap[uid] = u;
    }

    // Create Excel workbook
    const workbook = new ExcelJS.Workbook();
    const sheet = workbook.addWorksheet(`${gameId} Sessions`);

    sheet.columns = [
      { header: "Player", key: "player", width: 20 },
      { header: "State", key: "state", width: 15 },
      { header: "Level Reached", key: "level_reached", width: 15 },
      { header: "Total Actions", key: "total_actions", width: 15 },
      { header: "Total Time (s)", key: "total_time", width: 15 },
      { header: "Game Overs", key: "game_overs", width: 12 },
      { header: "Score", key: "score", width: 10 },
      { header: "Started At", key: "started_at", width: 22 },
      { header: "Ended At", key: "ended_at", width: 22 },
      { header: "Level Stats", key: "level_stats", width: 50 },
    ];

    // Style header row
    const headerRow = sheet.getRow(1);
    headerRow.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
    headerRow.fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "FF2563EB" },
    };
    headerRow.alignment = { horizontal: "center", vertical: "middle" };
    headerRow.eachCell((cell) => {
      cell.border = {
        top: { style: "thin", color: { argb: "FFE5E7EB" } },
        bottom: { style: "thin", color: { argb: "FFE5E7EB" } },
        left: { style: "thin", color: { argb: "FFE5E7EB" } },
        right: { style: "thin", color: { argb: "FFE5E7EB" } },
      };
    });

    // Data rows
    for (const s of sessions) {
      const user = s.user_id ? usersMap[s.user_id] : null;
      const player = s.player_name || (user ? user.username : "Anonymous");
      const levelStats = s.level_stats ?? [];

      const row = sheet.addRow({
        player,
        state: s.state || "NOT_FINISHED",
        level_reached: (s.current_level || 0) + 1,
        total_actions: s.total_actions || 0,
        total_time: Math.round((s.total_time || 0) * 100) / 100,
        game_overs: s.game_overs || 0,
        score: s.score || 0,
        started_at: s.started_at || "",
        ended_at: s.ended_at || "",
        level_stats: formatLevelStatsStr(levelStats),
      });

      row.eachCell((cell, colNumber) => {
        cell.border = {
          top: { style: "thin", color: { argb: "FFE5E7EB" } },
          bottom: { style: "thin", color: { argb: "FFE5E7EB" } },
          left: { style: "thin", color: { argb: "FFE5E7EB" } },
          right: { style: "thin", color: { argb: "FFE5E7EB" } },
        };
        // Center numeric columns: level_reached(3), total_actions(4), total_time(5), game_overs(6), score(7)
        if (colNumber >= 3 && colNumber <= 7) {
          cell.alignment = { horizontal: "center" };
        }
      });
    }

    // Summary sheet
    const summarySheet = workbook.addWorksheet("Summary");
    const totalSessions = sessions.length;
    const winCount = sessions.filter((s: any) => s.state === "WIN").length;
    const goCount = sessions.filter((s: any) => s.state === "GAME_OVER").length;
    const inProgressCount = sessions.filter((s: any) => s.state === "NOT_FINISHED").length;
    const totalDeaths = sessions.reduce((sum: number, s: any) => sum + (s.game_overs || 0), 0);
    const avgTime =
      totalSessions > 0
        ? Math.round(
            (sessions.reduce((sum: number, s: any) => sum + (s.total_time || 0), 0) / totalSessions) * 100,
          ) / 100
        : 0;
    const avgActions =
      totalSessions > 0
        ? Math.round(
            (sessions.reduce((sum: number, s: any) => sum + (s.total_actions || 0), 0) / totalSessions) * 10,
          ) / 10
        : 0;

    const summaryData: [string, string | number][] = [
      ["Game ID", gameId],
      ["Total Sessions", totalSessions],
      ["Wins", winCount],
      ["Game Overs", goCount],
      ["In Progress", inProgressCount],
      ["Total Game Overs (deaths)", totalDeaths],
      ["Avg Time (s)", avgTime],
      ["Avg Actions", avgActions],
      ["Filter", filter],
      ["Exported At", new Date().toISOString()],
    ];

    summaryData.forEach(([label, value], idx) => {
      const row = summarySheet.getRow(idx + 1);
      row.getCell(1).value = label;
      row.getCell(1).font = { bold: true };
      row.getCell(2).value = value;
    });

    summarySheet.getColumn(1).width = 25;
    summarySheet.getColumn(2).width = 30;

    // Build filename
    let filename = `${gameId}_sessions_${filter}`;
    if (filter === "date" && date) {
      filename += `_${date}`;
    } else if (filter === "range" && dateFrom && dateTo) {
      filename += `_${dateFrom}_to_${dateTo}`;
    }
    filename += ".xlsx";

    res.setHeader(
      "Content-Type",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    );
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);

    await workbook.xlsx.write(res);
    res.end();
  }),
);

// ──── 8. GET /export-all — Export all sessions as Excel ────
router.get(
  "/export-all",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (req, res) => {
    const filter = (req.query.filter as string) || "all";
    const date = req.query.date as string | undefined;
    const dateFrom = req.query.date_from as string | undefined;
    const dateTo = req.query.date_to as string | undefined;

    // Build query with date filters
    let sql = "SELECT * FROM play_sessions WHERE 1=1";
    const params: any[] = [];
    let paramIdx = 1;

    if (filter === "today") {
      sql += " AND started_at >= CURRENT_DATE";
    } else if (filter === "date" && date) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
        throw new AppError("Invalid date format. Use YYYY-MM-DD", 400);
      }
      sql += ` AND started_at >= $${paramIdx} AND started_at < ($${paramIdx + 1}::date + INTERVAL '1 day')`;
      params.push(date, date);
      paramIdx += 2;
    } else if (filter === "range" && dateFrom && dateTo) {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(dateFrom) || !/^\d{4}-\d{2}-\d{2}$/.test(dateTo)) {
        throw new AppError("Invalid date format. Use YYYY-MM-DD", 400);
      }
      sql += ` AND started_at >= $${paramIdx} AND started_at < ($${paramIdx + 1}::date + INTERVAL '1 day')`;
      params.push(dateFrom, dateTo);
      paramIdx += 2;
    }

    sql += " ORDER BY started_at DESC";

    const sessions = (await pool.query(sql, params)).rows;

    // Pre-fetch all games and users for performance
    const allGames: Record<string, any> = {};
    for (const g of (await queryAll("SELECT * FROM games", []))) {
      allGames[g.id] = g;
    }
    const allUsers: Record<string, any> = {};
    for (const u of (await queryAll("SELECT * FROM users", []))) {
      allUsers[u.id] = u;
    }

    const workbook = new ExcelJS.Workbook();
    const sheet = workbook.addWorksheet("All Sessions");

    sheet.columns = [
      { header: "Game ID", key: "game_id", width: 15 },
      { header: "Player", key: "player", width: 20 },
      { header: "State", key: "state", width: 15 },
      { header: "Level Reached", key: "level_reached", width: 15 },
      { header: "Total Actions", key: "total_actions", width: 15 },
      { header: "Total Time (s)", key: "total_time", width: 15 },
      { header: "Game Overs", key: "game_overs", width: 12 },
      { header: "Score", key: "score", width: 10 },
      { header: "Started At", key: "started_at", width: 22 },
      { header: "Ended At", key: "ended_at", width: 22 },
      { header: "Level Stats", key: "level_stats", width: 50 },
    ];

    // Style header row
    const headerRow = sheet.getRow(1);
    headerRow.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
    headerRow.fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "FF2563EB" },
    };
    headerRow.alignment = { horizontal: "center", vertical: "middle" };
    headerRow.eachCell((cell) => {
      cell.border = {
        top: { style: "thin", color: { argb: "FFE5E7EB" } },
        bottom: { style: "thin", color: { argb: "FFE5E7EB" } },
        left: { style: "thin", color: { argb: "FFE5E7EB" } },
        right: { style: "thin", color: { argb: "FFE5E7EB" } },
      };
    });

    // Data rows
    for (const s of sessions) {
      const game = allGames[s.game_id];
      const user = s.user_id ? allUsers[s.user_id] : null;
      const player = s.player_name || (user ? user.username : "Anonymous");
      const gameIdStr = game ? game.game_id : "unknown";
      const levelStats = s.level_stats ?? [];

      const row = sheet.addRow({
        game_id: gameIdStr,
        player,
        state: s.state || "NOT_FINISHED",
        level_reached: (s.current_level || 0) + 1,
        total_actions: s.total_actions || 0,
        total_time: Math.round((s.total_time || 0) * 100) / 100,
        game_overs: s.game_overs || 0,
        score: s.score || 0,
        started_at: s.started_at || "",
        ended_at: s.ended_at || "",
        level_stats: formatLevelStatsStr(levelStats),
      });

      row.eachCell((cell, colNumber) => {
        cell.border = {
          top: { style: "thin", color: { argb: "FFE5E7EB" } },
          bottom: { style: "thin", color: { argb: "FFE5E7EB" } },
          left: { style: "thin", color: { argb: "FFE5E7EB" } },
          right: { style: "thin", color: { argb: "FFE5E7EB" } },
        };
        // Center numeric columns: level_reached(4), total_actions(5), total_time(6), game_overs(7), score(8)
        if (colNumber >= 4 && colNumber <= 8) {
          cell.alignment = { horizontal: "center" };
        }
      });
    }

    // Summary sheet with per-game breakdown
    const summarySheet = workbook.addWorksheet("Summary");

    const gamesWithSessions: Record<string, { plays: number; wins: number; game_overs_total: number }> = {};
    for (const s of sessions) {
      const game = allGames[s.game_id];
      const gid = game ? game.game_id : "unknown";
      if (!gamesWithSessions[gid]) {
        gamesWithSessions[gid] = { plays: 0, wins: 0, game_overs_total: 0 };
      }
      gamesWithSessions[gid].plays += 1;
      if (s.state === "WIN") {
        gamesWithSessions[gid].wins += 1;
      }
      gamesWithSessions[gid].game_overs_total += s.game_overs || 0;
    }

    // Per-game header
    summarySheet.getCell("A1").value = "Game ID";
    summarySheet.getCell("A1").font = { bold: true };
    summarySheet.getCell("B1").value = "Sessions";
    summarySheet.getCell("B1").font = { bold: true };
    summarySheet.getCell("C1").value = "Wins";
    summarySheet.getCell("C1").font = { bold: true };
    summarySheet.getCell("D1").value = "Total Deaths";
    summarySheet.getCell("D1").font = { bold: true };

    let rowIdx = 2;
    for (const [gid, stats] of Object.entries(gamesWithSessions)) {
      summarySheet.getCell(`A${rowIdx}`).value = gid;
      summarySheet.getCell(`B${rowIdx}`).value = stats.plays;
      summarySheet.getCell(`C${rowIdx}`).value = stats.wins;
      summarySheet.getCell(`D${rowIdx}`).value = stats.game_overs_total;
      rowIdx++;
    }

    // Overall summary below the per-game table
    const offset = Object.keys(gamesWithSessions).length + 3;
    const overallData: [string, string | number][] = [
      ["Total Sessions", sessions.length],
      ["Total Games", Object.keys(gamesWithSessions).length],
      ["Filter", filter],
      ["Exported At", new Date().toISOString()],
    ];
    overallData.forEach(([label, value], idx) => {
      const r = summarySheet.getRow(offset + idx);
      r.getCell(1).value = label;
      r.getCell(1).font = { bold: true };
      r.getCell(2).value = value;
    });

    summarySheet.getColumn(1).width = 20;
    summarySheet.getColumn(2).width = 15;
    summarySheet.getColumn(3).width = 10;
    summarySheet.getColumn(4).width = 15;

    // Build filename
    let filename = `all_games_sessions_${filter}`;
    if (filter === "date" && date) {
      filename += `_${date}`;
    } else if (filter === "range" && dateFrom && dateTo) {
      filename += `_${dateFrom}_to_${dateTo}`;
    }
    filename += ".xlsx";

    res.setHeader(
      "Content-Type",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    );
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);

    await workbook.xlsx.write(res);
    res.end();
  }),
);

// ──── 9. GET /export-games — Export games list as Excel ────
router.get(
  "/export-games",
  authenticateToken,
  requireAdmin,
  asyncHandler(async (_req, res) => {
    const games = await queryAll(
      "SELECT * FROM games ORDER BY created_at DESC",
      [],
    );

    // Pre-fetch all users
    const allUsers: Record<string, any> = {};
    for (const u of (await queryAll("SELECT * FROM users", []))) {
      allUsers[u.id] = u;
    }

    const workbook = new ExcelJS.Workbook();
    const sheet = workbook.addWorksheet("Games");

    sheet.columns = [
      { header: "Game ID", key: "game_id", width: 15 },
      { header: "Name", key: "name", width: 20 },
      { header: "Status", key: "status", width: 12 },
      { header: "Version", key: "version", width: 10 },
      { header: "Game Code", key: "game_code", width: 15 },
      { header: "Description", key: "description", width: 30 },
      { header: "Game Rules", key: "game_rules", width: 30 },
      { header: "Owner", key: "owner", width: 15 },
      { header: "Drive Link", key: "drive_link", width: 30 },
      { header: "Video Link", key: "video_link", width: 30 },
      { header: "Default FPS", key: "default_fps", width: 12 },
      { header: "Tags", key: "tags", width: 20 },
      { header: "Levels", key: "levels", width: 8 },
      { header: "Total Plays", key: "total_plays", width: 12 },
      { header: "Total Wins", key: "total_wins", width: 12 },
      { header: "Avg Score", key: "avg_score", width: 12 },
      { header: "Uploaded By", key: "uploaded_by", width: 15 },
      { header: "Created At", key: "created_at", width: 22 },
    ];

    // Style header row
    const headerRow = sheet.getRow(1);
    headerRow.font = { bold: true, color: { argb: "FFFFFFFF" }, size: 11 };
    headerRow.fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "FF7C3AED" },
    };
    headerRow.alignment = { horizontal: "center", vertical: "middle" };
    headerRow.eachCell((cell) => {
      cell.border = {
        top: { style: "thin", color: { argb: "FFE5E7EB" } },
        bottom: { style: "thin", color: { argb: "FFE5E7EB" } },
        left: { style: "thin", color: { argb: "FFE5E7EB" } },
        right: { style: "thin", color: { argb: "FFE5E7EB" } },
      };
    });

    for (const g of games) {
      const uploader = g.uploaded_by ? allUsers[g.uploaded_by] : null;
      const baselineActions = g.baseline_actions ?? null;
      const numLevels = Array.isArray(baselineActions) ? baselineActions.length : 0;
      const tags = g.tags ?? [];
      const tagsStr = Array.isArray(tags) ? tags.join(", ") : "";

      const row = sheet.addRow({
        game_id: g.game_id,
        name: g.name,
        status: g.is_active ? "Active" : "Inactive",
        version: g.version,
        game_code: g.game_code,
        description: g.description || "",
        game_rules: g.game_rules || "",
        owner: g.game_owner_name || "",
        drive_link: g.game_drive_link || "",
        video_link: g.game_video_link || "",
        default_fps: g.default_fps || 5,
        tags: tagsStr,
        levels: numLevels,
        total_plays: g.total_plays || 0,
        total_wins: g.total_wins || 0,
        avg_score: Math.round((g.avg_score || 0) * 100) / 100,
        uploaded_by: uploader ? uploader.username : "",
        created_at: g.created_at || "",
      });

      row.eachCell((cell, colNumber) => {
        cell.border = {
          top: { style: "thin", color: { argb: "FFE5E7EB" } },
          bottom: { style: "thin", color: { argb: "FFE5E7EB" } },
          left: { style: "thin", color: { argb: "FFE5E7EB" } },
          right: { style: "thin", color: { argb: "FFE5E7EB" } },
        };
        // Center numeric columns: default_fps(11), levels(13), total_plays(14), total_wins(15), avg_score(16)
        if (colNumber === 11 || (colNumber >= 13 && colNumber <= 16)) {
          cell.alignment = { horizontal: "center" };
        }
      });
    }

    const timestamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 15).replace(/(\d{8})(\d{6})/, "$1_$2");
    const filename = `games_list_${timestamp}.xlsx`;

    res.setHeader(
      "Content-Type",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    );
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);

    await workbook.xlsx.write(res);
    res.end();
  }),
);

export default router;
