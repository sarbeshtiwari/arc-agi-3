import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './hooks/useAuth';

// Public pages
import HomePage from './pages/HomePage';
import PublicPlayPage from './pages/PublicPlayPage';
import DirectPlayPage from './pages/DirectPlayPage';

// Admin pages
import Layout from './components/Layout';
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

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!user) return <Navigate to="/admin/login" replace />;
  return children;
}

function PageGuard({ page, children }) {
  const { user } = useAuth();

  // Admins always have full access
  if (user?.is_admin) return children;

  const allowed = user?.allowed_pages || [];
  if (!allowed.includes(page)) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-full bg-red-500/10 border-2 border-red-500/30 flex items-center justify-center mb-4">
          <span className="text-2xl">🚫</span>
        </div>
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">Access Denied</h2>
        <p className="text-gray-500 dark:text-gray-400 text-sm">You don't have permission to access this page.</p>
        <p className="text-gray-400 dark:text-gray-500 text-xs mt-1">Contact an admin to request access.</p>
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

      {/* ── Admin routes (auth required) ── */}
      <Route path="/admin/login" element={<LoginPage />} />
      <Route path="/admin" element={
        <ProtectedRoute>
          <Layout />
        </ProtectedRoute>
      }>
        <Route index element={<PageGuard page="dashboard"><DashboardPage /></PageGuard>} />
        <Route path="games" element={<PageGuard page="games"><GamesPage /></PageGuard>} />
        <Route path="games/upload" element={<PageGuard page="upload"><GameUploadPage /></PageGuard>} />
        <Route path="games/:gameId" element={<PageGuard page="games"><GameDetailPage /></PageGuard>} />
        <Route path="games/:gameId/play" element={<PageGuard page="games"><GamePlayPage /></PageGuard>} />
        <Route path="requests" element={<PageGuard page="requests"><RequestedGamesPage /></PageGuard>} />
        <Route path="temp-games" element={<PageGuard page="games"><TempGamesPage /></PageGuard>} />
        <Route path="users" element={<PageGuard page="users"><UsersPage /></PageGuard>} />
        <Route path="eval" element={<PageGuard page="eval"><EvalPage /></PageGuard>} />
        <Route path="logs" element={<PageGuard page="logs"><LogsPage /></PageGuard>} />
        <Route path="system" element={<PageGuard page="logs"><SystemStatusPage /></PageGuard>} />
      </Route>

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
