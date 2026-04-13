import pool, { genId, queryOne, queryAll } from "../db.js";
import type { NotificationType, AuditAction } from "../../shared/types.js";
import type { Response } from "express";

const sseClients = new Map<string, Set<Response>>();

export function addSSEClient(userId: string, res: Response): void {
  if (!sseClients.has(userId)) sseClients.set(userId, new Set());
  sseClients.get(userId)!.add(res);
}

export function removeSSEClient(userId: string, res: Response): void {
  const clients = sseClients.get(userId);
  if (!clients) return;
  clients.delete(res);
  if (clients.size === 0) sseClients.delete(userId);
}

function emitToUser(userId: string, event: string, data: any): void {
  const clients = sseClients.get(userId);
  if (!clients || clients.size === 0) return;
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) {
    try { res.write(payload); } catch { clients.delete(res); }
  }
}

export async function createNotification(
  userId: string,
  type: NotificationType,
  title: string,
  message: string,
  gameId?: string | null,
): Promise<void> {
  await pool.query(
    `INSERT INTO notifications (id, user_id, type, title, message, game_id)
     VALUES ($1, $2, $3, $4, $5, $6)`,
    [genId(), userId, type, title, message, gameId || null],
  );
  emitToUser(userId, "notification", { type, title, message, game_id: gameId || null });
}

/** Send the same notification to multiple users. */
export async function notifyMany(
  userIds: string[],
  type: NotificationType,
  title: string,
  message: string,
  gameId?: string | null,
): Promise<void> {
  if (userIds.length === 0) return;
  const promises = userIds.map((uid) =>
    createNotification(uid, type, title, message, gameId),
  );
  await Promise.all(promises);
}

export async function getUserNotifications(
  userId: string,
  opts: { unreadOnly?: boolean; limit?: number } = {},
): Promise<any[]> {
  const { unreadOnly = false, limit = 50 } = opts;
  const conditions = ["n.user_id = $1"];
  const params: any[] = [userId];

  if (unreadOnly) {
    conditions.push("n.is_read = FALSE");
  }

  const rows = await queryAll(
    `SELECT n.*, g.game_id as game_code
     FROM notifications n
     LEFT JOIN games g ON g.id = n.game_id
     WHERE ${conditions.join(" AND ")}
     ORDER BY n.created_at DESC
     LIMIT $${params.length + 1}`,
    [...params, limit],
  );

  return rows.map((r) => ({
    id: r.id,
    user_id: r.user_id,
    type: r.type,
    title: r.title,
    message: r.message,
    game_id: r.game_id,
    game_code: r.game_code || null,
    is_read: r.is_read,
    created_at: r.created_at,
  }));
}

export async function markNotificationRead(notificationId: string, userId: string): Promise<boolean> {
  const result = await pool.query(
    "UPDATE notifications SET is_read = TRUE WHERE id = $1 AND user_id = $2",
    [notificationId, userId],
  );
  return (result.rowCount ?? 0) > 0;
}

export async function markAllNotificationsRead(userId: string): Promise<number> {
  const result = await pool.query(
    "UPDATE notifications SET is_read = TRUE WHERE user_id = $1 AND is_read = FALSE",
    [userId],
  );
  return result.rowCount ?? 0;
}

export async function getUnreadCount(userId: string): Promise<number> {
  const row = await queryOne(
    "SELECT COUNT(*)::int as count FROM notifications WHERE user_id = $1 AND is_read = FALSE",
    [userId],
  );
  return row?.count ?? 0;
}

// ──── Audit log helpers ────

export async function createAuditEntry(
  gameId: string,
  userId: string,
  action: AuditAction,
  details?: Record<string, any> | null,
): Promise<void> {
  await pool.query(
    `INSERT INTO audit_log (id, game_id, user_id, action, details)
     VALUES ($1, $2, $3, $4, $5)`,
    [genId(), gameId, userId, action, details ? JSON.stringify(details) : null],
  );
}

export async function getGameAuditLog(gameId: string): Promise<any[]> {
  const rows = await queryAll(
    `SELECT a.*, u.username
     FROM audit_log a
     LEFT JOIN users u ON u.id = a.user_id
     WHERE a.game_id = $1
     ORDER BY a.created_at DESC`,
    [gameId],
  );

  return rows.map((r) => ({
    id: r.id,
    game_id: r.game_id,
    user_id: r.user_id,
    username: r.username || "Unknown",
    action: r.action,
    details: r.details,
    created_at: r.created_at,
  }));
}
