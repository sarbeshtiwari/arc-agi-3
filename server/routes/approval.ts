import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken, requireRole } from "../middleware/auth.js";
import { AppError } from "../middleware/errorHandler.js";
import pool, { queryOne, queryAll } from "../db.js";
import { getUserLeadIds } from "../services/teamHelpers.js";
import {
  createNotification,
  notifyMany,
  createAuditEntry,
  getGameAuditLog,
} from "../services/NotificationService.js";

const router = Router();

function formatGameForApproval(row: any): any {
  if (!row) return null;
  return {
    id: row.id,
    game_id: row.game_id,
    name: row.name,
    description: row.description || null,
    game_rules: row.game_rules || null,
    game_owner_name: row.game_owner_name || null,
    version: row.version,
    game_code: row.game_code,
    is_active: row.is_active,
    default_fps: row.default_fps ?? 5,
    baseline_actions: row.baseline_actions ?? null,
    tags: row.tags ?? [],
    approval_status: row.approval_status || "draft",
    assigned_ql_id: row.assigned_ql_id || null,
    assigned_pl_id: row.assigned_pl_id || null,
    rejection_reason: row.rejection_reason || null,
    rejection_by: row.rejection_by || null,
    uploaded_by: row.uploaded_by || null,
    uploader_username: row.uploader_username || null,
    ql_username: row.ql_username || null,
    pl_username: row.pl_username || null,
    total_plays: row.total_plays ?? 0,
    total_wins: row.total_wins ?? 0,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

const GAME_JOIN_SQL = `
  SELECT g.*,
    uploader.username AS uploader_username,
    ql.username AS ql_username,
    pl.username AS pl_username
  FROM games g
  LEFT JOIN users uploader ON uploader.id = g.uploaded_by
  LEFT JOIN users ql ON ql.id = g.assigned_ql_id
  LEFT JOIN users pl ON pl.id = g.assigned_pl_id
`;

router.get(
  "/my-games",
  authenticateToken,
  requireRole("tasker"),
  asyncHandler(async (req, res) => {
    const rows = await queryAll(
      `${GAME_JOIN_SQL} WHERE g.uploaded_by = $1 ORDER BY g.updated_at DESC`,
      [req.user.id],
    );
    res.json(rows.map(formatGameForApproval));
  }),
);

router.get(
  "/ql-queue",
  authenticateToken,
  requireRole("ql"),
  asyncHandler(async (req, res) => {
    const rows = await queryAll(
      `${GAME_JOIN_SQL}
       WHERE g.assigned_ql_id = $1
       ORDER BY g.updated_at DESC`,
      [req.user.id],
    );
    res.json(rows.map(formatGameForApproval));
  }),
);

router.get(
  "/pl-queue",
  authenticateToken,
  requireRole("pl"),
  asyncHandler(async (req, res) => {
    const rows = await queryAll(
      `${GAME_JOIN_SQL}
       WHERE g.assigned_pl_id = $1
       ORDER BY g.updated_at DESC`,
      [req.user.id],
    );
    res.json(rows.map(formatGameForApproval));
  }),
);

router.get(
  "/all",
  authenticateToken,
  requireRole("super_admin"),
  asyncHandler(async (_req, res) => {
    const rows = await queryAll(
      `${GAME_JOIN_SQL} ORDER BY g.updated_at DESC`,
    );
    res.json(rows.map(formatGameForApproval));
  }),
);

router.post(
  "/:gameId/submit-for-review",
  authenticateToken,
  requireRole("tasker"),
  asyncHandler(async (req, res) => {
    const game = await queryOne("SELECT * FROM games WHERE game_id = $1 OR id = $1", [req.params.gameId]);
    if (!game) throw new AppError("Game not found", 404);

    if (game.uploaded_by !== req.user.id) {
      throw new AppError("You can only submit your own games", 403);
    }

    if (!["draft", "rejected"].includes(game.approval_status)) {
      throw new AppError(`Cannot submit a game with status '${game.approval_status}'`, 400);
    }

    const leadIds = await getUserLeadIds(req.user.id);
    const ql = leadIds.length > 0
      ? await queryOne("SELECT * FROM users WHERE id = ANY($1::text[]) AND role = 'ql' LIMIT 1", [leadIds])
      : null;
    if (!ql) {
      throw new AppError("No QL assigned to your team. Contact Super Admin.", 400);
    }

    const isResubmit = game.approval_status === "rejected";

    await pool.query(
      `UPDATE games SET
        approval_status = 'pending_ql',
        assigned_ql_id = $1,
        rejection_reason = NULL,
        rejection_by = NULL,
        updated_at = NOW()
       WHERE id = $2`,
      [ql.id, game.id],
    );

    const auditAction = isResubmit ? "game_resubmitted" as const : "game_submitted_for_review" as const;
    await createAuditEntry(game.id, req.user.id, auditAction, {
      message: req.body.message || null,
      ql_id: ql.id,
      ql_username: ql.username,
    });

    const notifType = isResubmit ? "game_resubmitted" as const : "game_submitted" as const;
    await createNotification(
      ql.id,
      notifType,
      isResubmit ? `Game resubmitted: ${game.game_id}` : `New game for review: ${game.game_id}`,
      req.body.message || `${req.user.username} ${isResubmit ? "resubmitted" : "submitted"} game '${game.game_id}' for review.`,
      game.id,
    );

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGameForApproval(updated));
  }),
);

router.post(
  "/:gameId/ql-review",
  authenticateToken,
  requireRole("ql"),
  asyncHandler(async (req, res) => {
    const { action, reason } = req.body;
    if (!action || !["approve", "reject"].includes(action)) {
      throw new AppError("action must be 'approve' or 'reject'", 400);
    }

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1 OR id = $1", [req.params.gameId]);
    if (!game) throw new AppError("Game not found", 404);

    if (game.assigned_ql_id !== req.user.id) {
      throw new AppError("This game is not assigned to you", 403);
    }

    if (game.approval_status !== "pending_ql") {
      throw new AppError(`Game is not pending QL review (status: ${game.approval_status})`, 400);
    }

    if (action === "approve") {
      const qlLeadIds = await getUserLeadIds(req.user.id);
      const pl = qlLeadIds.length > 0
        ? await queryOne("SELECT * FROM users WHERE id = ANY($1::text[]) AND role = 'pl' LIMIT 1", [qlLeadIds])
        : null;

      await pool.query(
        `UPDATE games SET
          approval_status = 'pending_pl',
          assigned_pl_id = $1,
          rejection_reason = NULL,
          rejection_by = NULL,
          updated_at = NOW()
         WHERE id = $2`,
        [pl?.id || null, game.id],
      );

      await createAuditEntry(game.id, req.user.id, "game_approved_by_ql", {
        message: reason || null,
        pl_id: pl?.id || null,
      });

      if (game.uploaded_by) {
        await createNotification(
          game.uploaded_by,
          "game_approved_ql",
          `Game approved by QL: ${game.game_id}`,
          `Your game '${game.game_id}' was approved by QL ${req.user.username}. Now pending PL review.`,
          game.id,
        );
      }

      if (pl) {
        await createNotification(
          pl.id,
          "game_submitted",
          `Game ready for PL review: ${game.game_id}`,
          `Game '${game.game_id}' was approved by QL ${req.user.username} and needs your final review.`,
          game.id,
        );
      }
    } else {
      if (!reason) throw new AppError("Rejection reason is required", 400);

      await pool.query(
        `UPDATE games SET
          approval_status = 'rejected',
          rejection_reason = $1,
          rejection_by = $2,
          updated_at = NOW()
         WHERE id = $3`,
        [reason, req.user.id, game.id],
      );

      await createAuditEntry(game.id, req.user.id, "game_rejected_by_ql", { reason });

      if (game.uploaded_by) {
        await createNotification(
          game.uploaded_by,
          "game_rejected",
          `Game rejected by QL: ${game.game_id}`,
          `Your game '${game.game_id}' was rejected by QL ${req.user.username}. Reason: ${reason}`,
          game.id,
        );
      }
    }

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGameForApproval(updated));
  }),
);

router.post(
  "/:gameId/pl-review",
  authenticateToken,
  requireRole("pl"),
  asyncHandler(async (req, res) => {
    const { action, reason } = req.body;
    if (!action || !["approve", "reject"].includes(action)) {
      throw new AppError("action must be 'approve' or 'reject'", 400);
    }

    const game = await queryOne("SELECT * FROM games WHERE game_id = $1 OR id = $1", [req.params.gameId]);
    if (!game) throw new AppError("Game not found", 404);

    if (game.approval_status !== "pending_pl") {
      throw new AppError(`Game is not pending PL review (status: ${game.approval_status})`, 400);
    }

    if (action === "approve") {
      await pool.query(
        `UPDATE games SET
          approval_status = 'approved',
          is_active = TRUE,
          rejection_reason = NULL,
          rejection_by = NULL,
          updated_at = NOW()
         WHERE id = $1`,
        [game.id],
      );

      await createAuditEntry(game.id, req.user.id, "game_approved_by_pl", {
        message: reason || null,
      });

      const notifyUserIds: string[] = [];
      if (game.uploaded_by) notifyUserIds.push(game.uploaded_by);
      if (game.assigned_ql_id) notifyUserIds.push(game.assigned_ql_id);

      await notifyMany(
        notifyUserIds,
        "game_approved_pl",
        `Game fully approved: ${game.game_id}`,
        `Game '${game.game_id}' was approved by PL ${req.user.username} and is now live!`,
        game.id,
      );
    } else {
      if (!reason) throw new AppError("Rejection reason is required", 400);

      await pool.query(
        `UPDATE games SET
          approval_status = 'rejected',
          rejection_reason = $1,
          rejection_by = $2,
          updated_at = NOW()
         WHERE id = $3`,
        [reason, req.user.id, game.id],
      );

      await createAuditEntry(game.id, req.user.id, "game_rejected_by_pl", { reason });

      const notifyUserIds: string[] = [];
      if (game.uploaded_by) notifyUserIds.push(game.uploaded_by);
      if (game.assigned_ql_id) notifyUserIds.push(game.assigned_ql_id);

      await notifyMany(
        notifyUserIds,
        "game_rejected",
        `Game rejected by PL: ${game.game_id}`,
        `Game '${game.game_id}' was rejected by PL ${req.user.username}. Reason: ${reason}`,
        game.id,
      );
    }

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGameForApproval(updated));
  }),
);

router.post(
  "/:gameId/admin-approve",
  authenticateToken,
  requireRole("super_admin"),
  asyncHandler(async (req, res) => {
    const game = await queryOne("SELECT * FROM games WHERE game_id = $1 OR id = $1", [req.params.gameId]);
    if (!game) throw new AppError("Game not found", 404);

    if (game.approval_status === "approved") {
      throw new AppError("Game is already approved", 400);
    }

    await pool.query(
      `UPDATE games SET
        approval_status = 'approved',
        is_active = TRUE,
        rejection_reason = NULL,
        rejection_by = NULL,
        updated_at = NOW()
       WHERE id = $1`,
      [game.id],
    );

    await createAuditEntry(game.id, req.user.id, "game_approved_by_admin", {
      previous_status: game.approval_status,
      message: req.body.message || null,
    });

    const notifyUserIds: string[] = [];
    if (game.uploaded_by) notifyUserIds.push(game.uploaded_by);
    if (game.assigned_ql_id) notifyUserIds.push(game.assigned_ql_id);

    await notifyMany(
      notifyUserIds,
      "game_approved_pl",
      `Game approved by Admin: ${game.game_id}`,
      `Game '${game.game_id}' was directly approved by Super Admin ${req.user.username} and is now live!`,
      game.id,
    );

    const updated = await queryOne("SELECT * FROM games WHERE id = $1", [game.id]);
    res.json(formatGameForApproval(updated));
  }),
);

router.get(
  "/:gameId/audit",
  authenticateToken,
  requireRole("tasker", "ql", "pl", "super_admin"),
  asyncHandler(async (req, res) => {
    const gameParam = req.params.gameId as string;
    const game = await queryOne(
      "SELECT * FROM games WHERE game_id = $1 OR id = $1",
      [gameParam],
    );
    if (!game) throw new AppError("Game not found", 404);

    const userRole = req.user.role;
    if (userRole === "tasker" && game.uploaded_by !== req.user.id) {
      throw new AppError("You can only view audit logs for your own games", 403);
    }
    if (userRole === "ql" && game.assigned_ql_id !== req.user.id && game.uploaded_by !== req.user.id) {
      throw new AppError("You can only view audit logs for games assigned to you", 403);
    }
    // super_admin and pl can view all audit logs

    const logs = await getGameAuditLog(game.id);
    res.json(logs);
  }),
);

export default router;
