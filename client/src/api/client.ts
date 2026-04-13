import axios from 'axios';

const API_BASE = '/api';

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// Request interceptor: attach JWT token
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('arc_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor: handle 401 — only redirect if on admin pages
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('arc_token');
      const path = window.location.pathname;
      if (path.startsWith('/admin') || path.startsWith('/dashboard')) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ──── Auth ────
export const authAPI = {
  login: (username, password) =>
    client.post('/auth/login', { username, password }),
  me: () => client.get('/auth/me'),
  register: (data) => client.post('/auth/register', data),
};

// ──── Games ────
export const gamesAPI = {
  // Public (no auth)
  listPublic: () => client.get('/games/public'),
  getPublic: (gameId) => client.get(`/games/public/${gameId}`),
  getPublicPlays: (gameId, limit = 20) =>
    client.get(`/games/public/${gameId}/plays`, { params: { limit } }),
  getPublicPreview: (gameId) =>
    client.get(`/games/public/${gameId}/preview`),
  getPublicStats: () =>
    client.get('/games/public/stats'),
  getPublicGameStats: (gameId) =>
    client.get(`/games/public/${gameId}/stats`),
  // Admin (auth required)
  list: (activeOnly = false) =>
    client.get('/games/', { params: { active_only: activeOnly } }),
  get: (gameId) => client.get(`/games/${gameId}`),
  upload: (formData) =>
    client.post('/games/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  update: (gameId, data) => client.put(`/games/${gameId}`, data),
  updateFiles: (gameId, formData) =>
    client.put(`/games/${gameId}/files`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  toggle: (gameId, isActive) =>
    client.patch(`/games/${gameId}/toggle`, { is_active: isActive }),
  delete: (gameId) => client.delete(`/games/${gameId}`),
  clearSessions: (gameId) => client.delete(`/games/${gameId}/sessions`),
  getSource: (gameId) => client.get(`/games/${gameId}/source`),
  syncLocal: () => client.post('/games/sync-local'),
  download: (gameId) => client.get(`/games/${gameId}/download`, { responseType: 'blob' }),
  bulkDownload: (gameIds: string[]) => client.post('/games/bulk-download', { game_ids: gameIds }, { responseType: 'blob' }),
  updateFiles: (gameId: string, formData: FormData) =>
    client.put(`/games/${gameId}/files`, formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  bulkUpload: (formData) =>
    client.post('/games/bulk-upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  // Videos
  // Video recording - streaming
  videoStart: (gameId, playerName = '', ext = 'webm') =>
    client.post(`/games/public/${gameId}/video/start`, { player_name: playerName, ext }),
  videoChunk: (gameId, recordingId, chunk) => {
    const fd = new FormData();
    fd.append('recording_id', recordingId);
    fd.append('chunk', chunk);
    return client.post(`/games/public/${gameId}/video/chunk`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  videoEnd: (gameId, recordingId) =>
    client.post(`/games/public/${gameId}/video/end`, { recording_id: recordingId }),
  // Video recording - legacy single upload (for download-then-upload)
  uploadVideo: (gameId, formData) =>
    client.post(`/games/public/${gameId}/video`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  listVideos: (gameId) => client.get(`/games/${gameId}/videos`),
  getVideoUrl: (gameId, filename) => `/api/games/${gameId}/videos/${filename}`,
  deleteVideo: (gameId, filename) => client.delete(`/games/${gameId}/videos/${filename}`),
};

// ──── Player ────
export const playerAPI = {
  // Public (no auth)
  publicStart: (gameId, seed = 0, playerName = null, startLevel = 0) =>
    client.post('/player/public/start', { game_id: gameId, seed, player_name: playerName, start_level: startLevel }),
  publicAction: (sessionGuid, action, x = null, y = null) =>
    client.post(`/player/public/action/${sessionGuid}`, { action, x, y }),
  publicEnd: (sessionGuid) =>
    client.post(`/player/public/end/${sessionGuid}`),
  // Ephemeral (temp game sessions, logged for 24hrs)
  ephemeralStart: (gameFile, metadataFile, playerName = '') => {
    const formData = new FormData();
    formData.append('game_file', gameFile);
    formData.append('metadata_file', metadataFile);
    formData.append('player_name', playerName || '');
    return client.post('/player/ephemeral/start', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  ephemeralAction: (sessionGuid, action, x = null, y = null) =>
    client.post(`/player/ephemeral/action/${sessionGuid}`, { action, x, y }),
  ephemeralEnd: (sessionGuid) =>
    client.post(`/player/ephemeral/end/${sessionGuid}`),
  // Admin (auth required)
  start: (gameId, seed = 0, startLevel = 0) =>
    client.post('/player/start', { game_id: gameId, seed, start_level: startLevel }),
  action: (sessionGuid, action, x = null, y = null) =>
    client.post(`/player/action/${sessionGuid}`, { action, x, y }),
  frame: (sessionGuid) => client.get(`/player/frame/${sessionGuid}`),
  end: (sessionGuid) => client.post(`/player/end/${sessionGuid}`),
  palette: () => client.get('/player/palette'),
};

// ──── Analytics ────
export const analyticsAPI = {
  dashboard: () => client.get('/analytics/dashboard'),
  game: (gameId, days = 30) =>
    client.get(`/analytics/game/${gameId}`, { params: { days } }),
  sessions: (gameId = null, limit = 20) =>
    client.get('/analytics/sessions', { params: { game_id: gameId, limit } }),
  replay: (sessionId) => client.get(`/analytics/replay/${sessionId}`),
  tempSessions: () => client.get('/analytics/temp-sessions'),
  deleteTempSessions: () => client.delete('/analytics/temp-sessions'),
  exportExcel: (gameId, params = {}) =>
    client.get(`/analytics/export/${gameId}`, { params, responseType: 'blob' }),
  exportAllExcel: (params = {}) =>
    client.get('/analytics/export-all', { params, responseType: 'blob' }),
  exportGamesList: () =>
    client.get('/analytics/export-games', { responseType: 'blob' }),
};

// ──── Users ────
export const usersAPI = {
  list: () => client.get('/users/'),
  get: (userId) => client.get(`/users/${userId}`),
  update: (userId, data) => client.put(`/users/${userId}`, data),
  delete: (userId) => client.delete(`/users/${userId}`),
  changeProtectedPassword: (secretCode, newPassword) =>
    client.post('/users/protected/change-password', { secret_code: secretCode, new_password: newPassword }),
  changeMyPassword: (currentPassword, newPassword) =>
    client.put('/users/me/change-password', { current_password: currentPassword, new_password: newPassword }),
};

// ──── Approval Workflow ────
export const approvalAPI = {
  myGames: () => client.get('/approval/my-games'),
  qlQueue: () => client.get('/approval/ql-queue'),
  plQueue: () => client.get('/approval/pl-queue'),
  allGames: () => client.get('/approval/all'),
  submitForReview: (gameId, message = '') =>
    client.post(`/approval/${gameId}/submit-for-review`, { message: message || null }),
  qlReview: (gameId, action, reason = null) =>
    client.post(`/approval/${gameId}/ql-review`, { action, reason }),
  plReview: (gameId, action, reason = null) =>
    client.post(`/approval/${gameId}/pl-review`, { action, reason }),
  adminApprove: (gameId, message = null) =>
    client.post(`/approval/${gameId}/admin-approve`, { message }),
  getAuditLog: (gameId) =>
    client.get(`/approval/${gameId}/audit`),
};

// ──── Teams ────
export const teamsAPI = {
  myTeam: () => client.get('/teams/my-team'),
  allUsers: () => client.get('/teams/all-users'),
  assign: (userId, leadId, action = 'add') =>
    client.put('/teams/assign', { user_id: userId, lead_id: leadId, action }),
  unassign: (userId, leadId) =>
    client.put('/teams/assign', { user_id: userId, lead_id: leadId, action: 'remove' }),
  unassigned: () => client.get('/teams/unassigned'),
  memberDetail: (userId, from = '', to = '') =>
    client.get(`/teams/${userId}/detail`, { params: { from: from || undefined, to: to || undefined } }),
  taskerDetail: (userId, from = '', to = '') =>
    client.get(`/teams/${userId}/tasker-detail`, { params: { from: from || undefined, to: to || undefined } }),
};

// ──── Notifications ────
export const notificationsAPI = {
  list: (unread = false, limit = 50) =>
    client.get('/notifications/', { params: { unread: unread || undefined, limit } }),
  unreadCount: () => client.get('/notifications/unread-count'),
  markRead: (id) => client.put(`/notifications/${id}/read`),
  markAllRead: () => client.put('/notifications/read-all'),
};

// ──── Game Requests ────
export const requestsAPI = {
  // Public
  submit: (formData) =>
    client.post('/requests/submit', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  // Admin
  list: (status = 'pending') =>
    client.get('/requests/', { params: { status } }),
  get: (requestId) => client.get(`/requests/${requestId}`),
  getSource: (requestId) => client.get(`/requests/${requestId}/source`),
  review: (requestId, action, adminNote = null) =>
    client.post(`/requests/${requestId}/review`, { action, admin_note: adminNote }),
  delete: (requestId) => client.delete(`/requests/${requestId}`),
  getFile: (requestId, fileType) =>
    client.get(`/requests/${requestId}/files/${fileType}`, { responseType: 'blob' }),
};

export const logsAPI = {
  list: (params: { level?: string; source?: string; search?: string; limit?: number; offset?: number; since?: string } = {}) =>
    client.get('/admin/logs', { params }),
  sources: () => client.get('/admin/logs/sources'),
  stats: () => client.get('/admin/logs/stats'),
  cleanup: () => client.delete('/admin/logs/cleanup'),
  streamUrl: () => {
    const token = localStorage.getItem('arc_token');
    return `${window.location.origin}/api/admin/logs/stream?token=${token}`;
  },
};

export const systemAPI = {
  status: () => client.get('/admin/system/status'),
  cleanupExpired: () => client.post('/admin/system/cleanup-expired'),
  killBridges: () => client.post('/admin/system/kill-bridges'),
};

export default client;
