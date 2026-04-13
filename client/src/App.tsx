import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './hooks/useAuth';

import HomePage from './pages/HomePage';
import PublicPlayPage from './pages/PublicPlayPage';
import DirectPlayPage from './pages/DirectPlayPage';

import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import GamesPage from './pages/GamesPage';
import GameDetailPage from './pages/GameDetailPage';
import GameUploadPage from './pages/GameUploadPage';
import GamePlayPage from './pages/GamePlayPage';
import UsersPage from './pages/UsersPage';
import RequestedGamesPage from './pages/RequestedGamesPage';
import TempGamesPage from './pages/TempGamesPage';
import EvalPage from './pages/EvalPage';
import LogsPage from './pages/LogsPage';
import SystemStatusPage from './pages/SystemStatusPage';

import DashboardLayout from './components/DashboardLayout';
import RoleDashboardPage from './pages/RoleDashboardPage';
import MyGamesPage from './pages/MyGamesPage';
import ReviewQueuePage from './pages/ReviewQueuePage';
import TeamManagementPage from './pages/TeamManagementPage';
import TeamMemberDetailPage from './pages/TeamMemberDetailPage';
import TaskerDetailPage from './pages/TaskerDetailPage';
import ProfilePage from './pages/ProfilePage';
import SettingsPage from './pages/SettingsPage';

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function RoleGuard({ roles, children }) {
  const { user } = useAuth();
  const userRole = user?.role || (user?.is_admin ? 'super_admin' : 'tasker');

  if (!roles.includes(userRole)) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-full bg-red-500/10 border-2 border-red-500/30 flex items-center justify-center mb-4">
          <span className="text-2xl">🚫</span>
        </div>
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">Access Denied</h2>
        <p className="text-gray-500 dark:text-gray-400 text-sm">This page is not available for your role.</p>
      </div>
    );
  }

  return children;
}

function AppRoutes() {
  return (
    <Routes>
      {/* ── Public routes (no auth) ── */}
      <Route path="/" element={<HomePage />} />
      <Route path="/play/:gameId" element={<PublicPlayPage />} />
      <Route path="/play-direct" element={<DirectPlayPage />} />

      {/* ── Shared login ── */}
      <Route path="/login" element={<LoginPage />} />

      {/* ── Unified dashboard (all authenticated users) ── */}
      <Route path="/dashboard" element={
        <ProtectedRoute>
          <DashboardLayout />
        </ProtectedRoute>
      }>
        <Route index element={<RoleDashboardPage />} />
        <Route path="my-games" element={<RoleGuard roles={['tasker', 'super_admin']}><MyGamesPage /></RoleGuard>} />
        <Route path="upload" element={<RoleGuard roles={['tasker', 'ql', 'pl', 'super_admin']}><GameUploadPage /></RoleGuard>} />
        <Route path="review" element={<RoleGuard roles={['ql', 'pl', 'super_admin']}><ReviewQueuePage /></RoleGuard>} />
        <Route path="team" element={<RoleGuard roles={['ql', 'pl', 'super_admin']}><TeamManagementPage /></RoleGuard>} />
        <Route path="team/:userId" element={<RoleGuard roles={['pl', 'super_admin']}><TeamMemberDetailPage /></RoleGuard>} />
        <Route path="team/tasker/:userId" element={<RoleGuard roles={['ql', 'pl', 'super_admin']}><TaskerDetailPage /></RoleGuard>} />
        <Route path="games/:gameId/play" element={<GamePlayPage />} />
        <Route path="profile" element={<ProfilePage />} />

        {/* ── Admin-only pages (super_admin) ── */}
        <Route path="games" element={<RoleGuard roles={['super_admin']}><GamesPage /></RoleGuard>} />
        <Route path="games/:gameId" element={<RoleGuard roles={['super_admin']}><GameDetailPage /></RoleGuard>} />
        <Route path="temp-games" element={<RoleGuard roles={['super_admin']}><TempGamesPage /></RoleGuard>} />
        <Route path="requests" element={<RoleGuard roles={['super_admin']}><RequestedGamesPage /></RoleGuard>} />
        <Route path="users" element={<RoleGuard roles={['super_admin']}><UsersPage /></RoleGuard>} />
        <Route path="settings" element={<RoleGuard roles={['super_admin']}><SettingsPage /></RoleGuard>} />
        <Route path="eval" element={<RoleGuard roles={['super_admin']}><EvalPage /></RoleGuard>} />
        <Route path="logs" element={<RoleGuard roles={['super_admin']}><LogsPage /></RoleGuard>} />
        <Route path="system" element={<RoleGuard roles={['super_admin']}><SystemStatusPage /></RoleGuard>} />
        <Route path="admin-stats" element={<RoleGuard roles={['super_admin']}><DashboardPage /></RoleGuard>} />
      </Route>

      {/* ── Backward compat: /admin/* → /dashboard ── */}
      <Route path="/admin/login" element={<Navigate to="/login" replace />} />
      <Route path="/admin/*" element={<Navigate to="/dashboard" replace />} />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}
