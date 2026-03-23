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
      if (window.location.pathname.startsWith('/admin')) {
        window.location.href = '/admin/login';
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

export default client;
