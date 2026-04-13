import { Router } from "express";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { authenticateToken } from "../middleware/auth.js";
import {
  getUserNotifications,
  markNotificationRead,
  markAllNotificationsRead,
  getUnreadCount,
  addSSEClient,
  removeSSEClient,
} from "../services/NotificationService.js";

const router = Router();

router.get(
  "/stream",
  authenticateToken,
  (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    res.write(`data: ${JSON.stringify({ connected: true })}\n\n`);

    const userId = req.user.id;
    addSSEClient(userId, res);

    const heartbeat = setInterval(() => {
      try { res.write(": heartbeat\n\n"); } catch {}
    }, 30000);

    req.on("close", () => {
      clearInterval(heartbeat);
      removeSSEClient(userId, res);
    });
  },
);

router.get(
  "/",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const unreadOnly = req.query.unread === "true";
    const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 50, 1), 200);

    const notifications = await getUserNotifications(req.user.id, { unreadOnly, limit });
    const unreadCount = await getUnreadCount(req.user.id);

    res.json({ notifications, unread_count: unreadCount });
  }),
);

router.get(
  "/unread-count",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const count = await getUnreadCount(req.user.id);
    res.json({ unread_count: count });
  }),
);

router.put(
  "/:id/read",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const success = await markNotificationRead(req.params.id as string, req.user.id);
    if (!success) {
      res.status(404).json({ detail: "Notification not found" });
      return;
    }
    res.json({ ok: true });
  }),
);

router.put(
  "/read-all",
  authenticateToken,
  asyncHandler(async (req, res) => {
    const count = await markAllNotificationsRead(req.user.id);
    res.json({ marked_read: count });
  }),
);

export default router;
