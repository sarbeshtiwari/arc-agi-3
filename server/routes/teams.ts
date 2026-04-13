import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireRole } from "../middleware/auth.js";
import { AppError } from "../middleware/errorHandler.js";
import pool, { genId, queryOne, queryAll } from "../db.js";
import { createNotification } from "../services/NotificationService.js";

const router = Router();

async function getLeadInfo(userId: string): Promise<{ ids: string[]; usernames: string[] }> {
  const rows = await queryAll(
    `SELECT utl.lead_id, u.username FROM user_team_leads utl
     JOIN users u ON u.id = utl.lead_id
     WHERE utl.user_id = $1 ORDER BY u.username`,
    [userId],
  );
  return {
    ids: rows.map((r: any) => r.lead_id),
    usernames: rows.map((r: any) => r.username),
  };
}

async function formatMemberWithLeads(row: any) {
  const leads = await getLeadInfo(row.id);
  return {
    id: row.id,
    username: row.username,
    display_name: row.display_name || null,
    email: row.email || null,
    role: row.role || "tasker",
    is_active: row.is_active,
    team_lead_id: row.team_lead_id || null,
    team_lead_ids: leads.ids,
    team_lead_usernames: leads.usernames,
    team_lead_username: row.team_lead_username || leads.usernames[0] || null,
    created_at: row.created_at,
  };
}

router.get(
  "/my-team",
  authenticateToken,
  requireRole("ql", "pl"),
  asyncHandler(async (req, res) => {
    const userRole = req.user.role;

    if (userRole === "ql") {
      const members = await queryAll(
        `SELECT DISTINCT u.*
         FROM users u
         JOIN user_team_leads utl ON utl.user_id = u.id AND utl.lead_id = $1
         ORDER BY u.username`,
        [req.user.id],
      );
      const result = await Promise.all(members.map(formatMemberWithLeads));
      res.json(result);
    } else if (userRole === "pl") {
      const qls = await queryAll(
        `SELECT DISTINCT u.*
         FROM users u
         JOIN user_team_leads utl ON utl.user_id = u.id AND utl.lead_id = $1
         WHERE u.role = 'ql'
         ORDER BY u.username`,
        [req.user.id],
      );

      const qlIds = qls.map((q: any) => q.id);
      let taskers: any[] = [];
      if (qlIds.length > 0) {
        const placeholders = qlIds.map((_: any, i: number) => `$${i + 1}`).join(", ");
        taskers = await queryAll(
          `SELECT DISTINCT u.*
           FROM users u
           JOIN user_team_leads utl ON utl.user_id = u.id
           WHERE utl.lead_id IN (${placeholders})
           ORDER BY u.username`,
          qlIds,
        );
      }

      res.json({
        qls: await Promise.all(qls.map(formatMemberWithLeads)),
        taskers: await Promise.all(taskers.map(formatMemberWithLeads)),
      });
    }
  }),
);

router.get(
  "/all-users",
  authenticateToken,
  requireRole("super_admin"),
  asyncHandler(async (_req, res) => {
    const users = await queryAll(
      `SELECT u.*, tl.username AS team_lead_username
       FROM users u
       LEFT JOIN users tl ON tl.id = u.team_lead_id
       ORDER BY u.role, u.username`,
    );
    const result = await Promise.all(users.map(formatMemberWithLeads));
    res.json(result);
  }),
);

router.put(
  "/assign",
  authenticateToken,
  requireRole("super_admin", "pl", "ql"),
  asyncHandler(async (req, res) => {
    const { user_id, lead_id, action: assignAction } = req.body;
    if (!user_id) throw new AppError("user_id is required", 400);

    const targetUser = await queryOne("SELECT * FROM users WHERE id = $1", [user_id]);
    if (!targetUser) throw new AppError("User not found", 404);

    const callerRole = req.user.role;

    if (assignAction === "remove") {
      if (!lead_id) throw new AppError("lead_id is required for remove action", 400);

      if (callerRole === "ql" && lead_id !== req.user.id) {
        throw new AppError("QLs can only remove from their own team", 403);
      }
      if (callerRole === "pl") {
        const isMyQL = await queryOne(
          `SELECT 1 FROM user_team_leads WHERE user_id = $1 AND lead_id = $2`,
          [lead_id, req.user.id],
        );
        if (!isMyQL && lead_id !== req.user.id) {
          throw new AppError("PLs can only manage their own QLs and those QLs' taskers", 403);
        }
      }

      await pool.query(
        `DELETE FROM user_team_leads WHERE user_id = $1 AND lead_id = $2`,
        [user_id, lead_id],
      );
      const remaining = await queryAll(
        `SELECT lead_id FROM user_team_leads WHERE user_id = $1 LIMIT 1`,
        [user_id],
      );
      const newPrimaryLead = remaining.length > 0 ? remaining[0].lead_id : null;
      await pool.query(
        "UPDATE users SET team_lead_id = $1, updated_at = NOW() WHERE id = $2",
        [newPrimaryLead, user_id],
      );
      await createNotification(
        user_id, "team_assigned", "Team assignment updated",
        `You have been removed from a team lead's group.`,
      );
    } else {
      if (!lead_id) throw new AppError("lead_id is required", 400);

      if (callerRole === "ql") {
        if (targetUser.role !== "tasker") {
          throw new AppError("QLs can only manage Taskers", 403);
        }
        await pool.query(
          `INSERT INTO user_team_leads (id, user_id, lead_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
          [genId(), user_id, req.user.id],
        );
        const existing = await queryOne(
          `SELECT team_lead_id FROM users WHERE id = $1`, [user_id],
        );
        if (!existing.team_lead_id) {
          await pool.query(
            "UPDATE users SET team_lead_id = $1, updated_at = NOW() WHERE id = $2",
            [req.user.id, user_id],
          );
        }
        await createNotification(
          user_id, "team_assigned", "Team assignment updated",
          `You have been assigned to QL ${req.user.username}'s team.`,
        );
      } else if (callerRole === "pl") {
        const lead = await queryOne("SELECT * FROM users WHERE id = $1", [lead_id]);
        if (!lead) throw new AppError("Team lead not found", 404);

        if (targetUser.role === "tasker" && lead.role !== "ql") {
          throw new AppError("Taskers must be assigned to a QL", 400);
        }
        if (targetUser.role === "ql" && lead_id !== req.user.id) {
          throw new AppError("PLs can only assign QLs to themselves", 400);
        }

        const isMyQL = lead.role === "ql" ? await queryOne(
          `SELECT 1 FROM user_team_leads WHERE user_id = $1 AND lead_id = $2`,
          [lead_id, req.user.id],
        ) : null;

        if (targetUser.role === "tasker" && lead.role === "ql" && !isMyQL) {
          throw new AppError("Can only assign taskers to your own QLs", 403);
        }

        await pool.query(
          `INSERT INTO user_team_leads (id, user_id, lead_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
          [genId(), user_id, lead_id],
        );
        const existing = await queryOne(
          `SELECT team_lead_id FROM users WHERE id = $1`, [user_id],
        );
        if (!existing.team_lead_id) {
          await pool.query(
            "UPDATE users SET team_lead_id = $1, updated_at = NOW() WHERE id = $2",
            [lead_id, user_id],
          );
        }
        await createNotification(
          user_id, "team_assigned", "Team assignment updated",
          `You have been assigned to ${lead.username}'s team by PL ${req.user.username}.`,
        );
      } else {
        const lead = await queryOne("SELECT * FROM users WHERE id = $1", [lead_id]);
        if (!lead) throw new AppError("Team lead not found", 404);

        if (targetUser.role === "tasker" && lead.role !== "ql") {
          throw new AppError("Taskers must be assigned to a QL", 400);
        }
        if (targetUser.role === "ql" && lead.role !== "pl") {
          throw new AppError("QLs must be assigned to a PL", 400);
        }

        await pool.query(
          `INSERT INTO user_team_leads (id, user_id, lead_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
          [genId(), user_id, lead_id],
        );
        const existing = await queryOne(
          `SELECT team_lead_id FROM users WHERE id = $1`, [user_id],
        );
        if (!existing.team_lead_id) {
          await pool.query(
            "UPDATE users SET team_lead_id = $1, updated_at = NOW() WHERE id = $2",
            [lead_id, user_id],
          );
        }
        const leadName = lead.username;
        await createNotification(
          user_id, "team_assigned", "Team assignment updated",
          `You have been assigned to ${leadName}'s team by Super Admin.`,
        );
      }
    }

    const updated = await queryOne("SELECT * FROM users WHERE id = $1", [user_id]);
    const result = await formatMemberWithLeads(updated);
    res.json(result);
  }),
);

// ────────────────────────────────────────────────
// Team member detail — taskers, their games, approval stats
// Accessible by PL (for their QLs) and Super Admin (for anyone)
// ────────────────────────────────────────────────
router.get(
  "/:userId/detail",
  authenticateToken,
  requireRole("pl", "super_admin"),
  asyncHandler(async (req, res) => {
    const targetId = req.params.userId as string;

    const targetUser = await queryOne("SELECT * FROM users WHERE id = $1", [targetId]);
    if (!targetUser) throw new AppError("User not found", 404);

    const callerRole = req.user.role;

    // PL can only view their own QLs
    if (callerRole === "pl") {
      const isMyQL = await queryOne(
        `SELECT 1 FROM user_team_leads WHERE user_id = $1 AND lead_id = $2`,
        [targetId, req.user.id],
      );
      if (!isMyQL) throw new AppError("This user is not in your team", 403);
    }

    // Get taskers assigned to this user (via junction table)
    const taskers = await queryAll(
      `SELECT u.* FROM users u
       JOIN user_team_leads utl ON utl.user_id = u.id AND utl.lead_id = $1
       ORDER BY u.username`,
      [targetId],
    );
    const taskerList = await Promise.all(taskers.map(formatMemberWithLeads));

    // Get all tasker IDs (including the target user themselves for their own uploads)
    const relevantUserIds = [targetId, ...taskers.map((t: any) => t.id)];

    // Date filters
    const { from, to } = req.query as { from?: string; to?: string };
    let dateFilter = "";
    const params: any[] = [relevantUserIds];
    let paramIndex = 2;

    if (from) {
      dateFilter += ` AND g.created_at >= $${paramIndex}`;
      params.push(from);
      paramIndex++;
    }
    if (to) {
      dateFilter += ` AND g.created_at <= $${paramIndex}`;
      params.push(to + "T23:59:59.999Z");
      paramIndex++;
    }

    // Get games uploaded by any of the relevant users
    const games = await queryAll(
      `SELECT g.*,
         uploader.username AS uploader_username,
         uploader.display_name AS uploader_display_name,
         ql.username AS ql_username,
         pl.username AS pl_username
       FROM games g
       LEFT JOIN users uploader ON uploader.id = g.uploaded_by
       LEFT JOIN users ql ON ql.id = g.assigned_ql_id
       LEFT JOIN users pl ON pl.id = g.assigned_pl_id
       WHERE g.uploaded_by = ANY($1::text[])
       ${dateFilter}
       ORDER BY g.updated_at DESC`,
      params,
    );

    // Compute stats
    const stats = {
      total: games.length,
      draft: 0,
      pending_ql: 0,
      ql_approved: 0,
      pending_pl: 0,
      approved: 0,
      rejected: 0,
    };
    for (const g of games) {
      const s = g.approval_status || "draft";
      if (s in stats) stats[s as keyof typeof stats]++;
    }

    const userInfo = await formatMemberWithLeads(targetUser);

    // For each game, get gameplay stats of the uploader
    const gamesWithStats = await Promise.all(
      games.map(async (g: any) => {
        const playStats = await queryOne(
          `SELECT
             COUNT(*)::int AS play_count,
             COUNT(*) FILTER (WHERE state = 'WIN')::int AS wins,
             COUNT(*) FILTER (WHERE state = 'GAME_OVER')::int AS losses,
             COALESCE(SUM(game_overs), 0)::int AS total_game_overs,
             COALESCE(SUM(total_actions), 0)::int AS total_actions,
             ROUND(COALESCE(SUM(total_time), 0)::numeric, 1) AS total_time_secs,
             MAX(COALESCE(ended_at, started_at)) AS last_played_at,
             MAX(current_level)::int AS max_level_reached
           FROM play_sessions
           WHERE game_id = $1 AND user_id = $2`,
          [g.id, g.uploaded_by],
        );

        const levelStatsRows = await queryAll(
          `SELECT level_stats FROM play_sessions
           WHERE game_id = $1 AND user_id = $2 AND level_stats IS NOT NULL AND level_stats != '[]'::jsonb`,
          [g.id, g.uploaded_by],
        );
        let levelsCleared = 0;
        for (const row of levelStatsRows) {
          const stats = Array.isArray(row.level_stats) ? row.level_stats : [];
          levelsCleared += stats.filter((ls: any) => ls.completed).length;
        }

        return {
          id: g.id,
          game_id: g.game_id,
          name: g.name,
          description: g.description,
          game_code: g.game_code,
          version: g.version,
          is_active: g.is_active,
          approval_status: g.approval_status || "draft",
          rejection_reason: g.rejection_reason,
          uploaded_by: g.uploaded_by,
          uploader_username: g.uploader_username,
          uploader_display_name: g.uploader_display_name,
          ql_username: g.ql_username,
          pl_username: g.pl_username,
          total_plays: g.total_plays ?? 0,
          total_wins: g.total_wins ?? 0,
          created_at: g.created_at,
          updated_at: g.updated_at,
          gameplay: {
            play_count: playStats?.play_count ?? 0,
            wins: playStats?.wins ?? 0,
            losses: playStats?.losses ?? 0,
            total_game_overs: playStats?.total_game_overs ?? 0,
            total_actions: playStats?.total_actions ?? 0,
            total_time_secs: parseFloat(playStats?.total_time_secs ?? "0"),
            last_played_at: playStats?.last_played_at || null,
            max_level_reached: playStats?.max_level_reached ?? 0,
            levels_cleared: levelsCleared,
          },
        };
      }),
    );

    res.json({
      user: userInfo,
      taskers: taskerList,
      games: gamesWithStats,
      stats,
    });
  }),
);

// ────────────────────────────────────────────────
// Tasker detail — individual tasker's games, history, files, audit trail
// Accessible by QL (for their taskers), PL (for their chain), Super Admin (anyone)
// ────────────────────────────────────────────────
router.get(
  "/:userId/tasker-detail",
  authenticateToken,
  requireRole("ql", "pl", "super_admin"),
  asyncHandler(async (req, res) => {
    const targetId = req.params.userId as string;

    const targetUser = await queryOne("SELECT * FROM users WHERE id = $1", [targetId]);
    if (!targetUser) throw new AppError("User not found", 404);

    const callerRole = req.user.role;

    // QL can only view their own taskers
    if (callerRole === "ql") {
      const isMine = await queryOne(
        `SELECT 1 FROM user_team_leads WHERE user_id = $1 AND lead_id = $2`,
        [targetId, req.user.id],
      );
      if (!isMine) throw new AppError("This user is not in your team", 403);
    }

    // PL can only view taskers in their QL chain
    if (callerRole === "pl") {
      // Get PLs QLs first
      const myQLs = await queryAll(
        `SELECT u.id FROM users u
         JOIN user_team_leads utl ON utl.user_id = u.id AND utl.lead_id = $1
         WHERE u.role = 'ql'`,
        [req.user.id],
      );
      const qlIds = myQLs.map((q: any) => q.id);

      // Check if target is either a QL of this PL or a tasker of one of their QLs
      const isMyQL = qlIds.includes(targetId);
      let isMyTasker = false;
      if (!isMyQL && qlIds.length > 0) {
        const placeholders = qlIds.map((_: any, i: number) => `$${i + 1}`).join(", ");
        const check = await queryOne(
          `SELECT 1 FROM user_team_leads WHERE user_id = $${qlIds.length + 1} AND lead_id IN (${placeholders})`,
          [...qlIds, targetId],
        );
        isMyTasker = !!check;
      }

      if (!isMyQL && !isMyTasker) {
        throw new AppError("This user is not in your team chain", 403);
      }
    }

    const userInfo = await formatMemberWithLeads(targetUser);

    // Date filters
    const { from, to } = req.query as { from?: string; to?: string };
    let dateFilter = "";
    const params: any[] = [targetId];
    let paramIndex = 2;

    if (from) {
      dateFilter += ` AND g.created_at >= $${paramIndex}`;
      params.push(from);
      paramIndex++;
    }
    if (to) {
      dateFilter += ` AND g.created_at <= $${paramIndex}`;
      params.push(to + "T23:59:59.999Z");
      paramIndex++;
    }

    // Get all games uploaded by this user with full details
    const games = await queryAll(
      `SELECT g.*,
         uploader.username AS uploader_username,
         uploader.display_name AS uploader_display_name,
         ql_user.username AS ql_username,
         pl_user.username AS pl_username
       FROM games g
       LEFT JOIN users uploader ON uploader.id = g.uploaded_by
       LEFT JOIN users ql_user ON ql_user.id = g.assigned_ql_id
       LEFT JOIN users pl_user ON pl_user.id = g.assigned_pl_id
       WHERE g.uploaded_by = $1
       ${dateFilter}
       ORDER BY g.updated_at DESC`,
      params,
    );

    // For each game, get audit log + gameplay stats for this user
    const gamesWithHistory = await Promise.all(
      games.map(async (g: any) => {
        const [auditLogs, playStats] = await Promise.all([
          queryAll(
            `SELECT a.*, u.username
             FROM audit_log a
             LEFT JOIN users u ON u.id = a.user_id
             WHERE a.game_id = $1
             ORDER BY a.created_at ASC`,
            [g.id],
          ),
          queryOne(
            `SELECT
               COUNT(*)::int AS play_count,
               COUNT(*) FILTER (WHERE state = 'WIN')::int AS wins,
               COUNT(*) FILTER (WHERE state = 'GAME_OVER')::int AS losses,
               COALESCE(SUM(game_overs), 0)::int AS total_game_overs,
               COALESCE(SUM(total_actions), 0)::int AS total_actions,
               ROUND(COALESCE(SUM(total_time), 0)::numeric, 1) AS total_time_secs,
               ROUND(COALESCE(AVG(total_time), 0)::numeric, 1) AS avg_time_secs,
               MAX(COALESCE(ended_at, started_at)) AS last_played_at,
               MAX(current_level)::int AS max_level_reached
             FROM play_sessions
             WHERE game_id = $1 AND user_id = $2`,
            [g.id, targetId],
          ),
        ]);

        // Count total levels cleared from level_stats JSONB
        const levelStatsRows = await queryAll(
          `SELECT level_stats FROM play_sessions
           WHERE game_id = $1 AND user_id = $2 AND level_stats IS NOT NULL AND level_stats != '[]'::jsonb`,
          [g.id, targetId],
        );
        let totalLevelsCleared = 0;
        for (const row of levelStatsRows) {
          const stats = Array.isArray(row.level_stats) ? row.level_stats : [];
          totalLevelsCleared += stats.filter((ls: any) => ls.completed).length;
        }

        return {
          id: g.id,
          game_id: g.game_id,
          name: g.name,
          description: g.description,
          game_code: g.game_code,
          version: g.version,
          is_active: g.is_active,
          approval_status: g.approval_status || "draft",
          rejection_reason: g.rejection_reason,
          uploaded_by: g.uploaded_by,
          uploader_username: g.uploader_username,
          uploader_display_name: g.uploader_display_name,
          ql_username: g.ql_username,
          pl_username: g.pl_username,
          total_plays: g.total_plays ?? 0,
          total_wins: g.total_wins ?? 0,
          default_fps: g.default_fps ?? 5,
          has_game_file: !!g.game_file_path,
          has_metadata_file: !!g.metadata_file_path,
          created_at: g.created_at,
          updated_at: g.updated_at,
          // History / timeline
          update_count: auditLogs.length,
          first_update: auditLogs.length > 0 ? auditLogs[0].created_at : g.created_at,
          last_update: auditLogs.length > 0 ? auditLogs[auditLogs.length - 1].created_at : g.updated_at,
          audit_log: auditLogs.map((a: any) => ({
            id: a.id,
            action: a.action,
            username: a.username || "Unknown",
            details: a.details,
            created_at: a.created_at,
          })),
          // Gameplay stats for this user
          gameplay: {
            play_count: playStats?.play_count ?? 0,
            wins: playStats?.wins ?? 0,
            losses: playStats?.losses ?? 0,
            total_game_overs: playStats?.total_game_overs ?? 0,
            total_actions: playStats?.total_actions ?? 0,
            total_time_secs: parseFloat(playStats?.total_time_secs ?? "0"),
            avg_time_secs: parseFloat(playStats?.avg_time_secs ?? "0"),
            last_played_at: playStats?.last_played_at || null,
            max_level_reached: playStats?.max_level_reached ?? 0,
            levels_cleared: totalLevelsCleared,
          },
        };
      }),
    );

    // Compute stats
    const stats = {
      total: games.length,
      draft: 0,
      pending_ql: 0,
      ql_approved: 0,
      pending_pl: 0,
      approved: 0,
      rejected: 0,
    };
    for (const g of games) {
      const s = g.approval_status || "draft";
      if (s in stats) stats[s as keyof typeof stats]++;
    }

    res.json({
      user: userInfo,
      games: gamesWithHistory,
      stats,
    });
  }),
);

router.get(
  "/unassigned",
  authenticateToken,
  requireRole("super_admin", "pl", "ql"),
  asyncHandler(async (req, res) => {
    const callerRole = req.user.role;

    let rows: any[];
    if (callerRole === "ql") {
      rows = await queryAll(
        `SELECT u.*
         FROM users u
         WHERE u.role = 'tasker' AND u.is_active = TRUE
           AND NOT EXISTS (SELECT 1 FROM user_team_leads utl WHERE utl.user_id = u.id)
         ORDER BY u.username`,
      );
    } else if (callerRole === "pl") {
      rows = await queryAll(
        `SELECT u.*
         FROM users u
         WHERE u.role IN ('tasker', 'ql') AND u.is_active = TRUE
           AND NOT EXISTS (SELECT 1 FROM user_team_leads utl WHERE utl.user_id = u.id)
         ORDER BY u.role, u.username`,
      );
    } else {
      rows = await queryAll(
        `SELECT u.*
         FROM users u
         WHERE u.role IN ('tasker', 'ql') AND u.is_active = TRUE
           AND NOT EXISTS (SELECT 1 FROM user_team_leads utl WHERE utl.user_id = u.id)
         ORDER BY u.role, u.username`,
      );
    }

    const result = await Promise.all(rows.map(formatMemberWithLeads));
    res.json(result);
  }),
);

export default router;
