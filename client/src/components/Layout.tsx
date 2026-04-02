import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../hooks/useAuth';
import {
  LayoutDashboard, Gamepad2, Upload, Users, LogOut,
  ChevronLeft, ChevronRight, Inbox, ExternalLink, Timer, FlaskConical, ScrollText,
  Sun, Moon, Activity
} from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

const navSections = [
  {
    label: 'Overview',
    items: [
      { path: '/admin', label: 'Dashboard', icon: LayoutDashboard, exact: true, page: 'dashboard' },
    ],
  },
  {
    label: 'Games',
    items: [
      { path: '/admin/games', label: 'All Games', icon: Gamepad2, page: 'games' },
      { path: '/admin/temp-games', label: 'Temp Games', icon: Timer, page: 'games' },
      { path: '/admin/games/upload', label: 'Upload', icon: Upload, page: 'upload' },
      { path: '/admin/requests', label: 'Requests', icon: Inbox, page: 'requests' },
    ],
  },
  {
    label: 'Team',
    items: [
      { path: '/admin/users', label: 'Users', icon: Users, page: 'users' },
    ],
  },
  {
    label: 'AI',
    items: [
      { path: '/admin/eval', label: 'Eval Runner', icon: FlaskConical, page: 'eval' },
    ],
  },
  {
    label: 'System',
    items: [
      { path: '/admin/logs', label: 'Logs', icon: ScrollText, page: 'logs' },
      { path: '/admin/system', label: 'System Status', icon: Activity, page: 'logs' },
    ],
  },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const { theme, toggleTheme } = useTheme();

  // Filter nav items based on user's allowed_pages
  const allowedPages = user?.allowed_pages || [];
  const isAllowed = (page) => user?.is_admin || allowedPages.includes(page);

  const filteredSections = navSections
    .map(section => ({
      ...section,
      items: section.items.filter(item => isAllowed(item.page)),
    }))
    .filter(section => section.items.length > 0);

  const isActive = (item) => {
    if (item.exact) return location.pathname === item.path;
    return location.pathname === item.path ||
      (item.path !== '/admin' && location.pathname.startsWith(item.path));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex">
      {/* Sidebar */}
      <motion.aside
        className={`bg-white dark:bg-gray-950 border-r border-gray-200 dark:border-gray-800/60 flex flex-col shrink-0 relative h-screen sticky top-0`}
        animate={{ width: collapsed ? 68 : 240 }}
        transition={{ duration: 0.2, ease: 'easeInOut' }}
      >
        {/* Header */}
        <div className={`h-14 flex items-center border-b border-gray-200 dark:border-gray-800/60 ${collapsed ? 'justify-center px-0' : 'px-5'}`}>
          {!collapsed && (
            <Link to="/admin" className="flex items-center gap-2.5 group">
              <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
                <span className="text-white text-xs font-bold">A</span>
              </div>
              <div>
                <p className="text-sm font-semibold text-gray-900 dark:text-white leading-none">ARC-AGI</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 leading-none mt-0.5">Admin Panel</p>
              </div>
            </Link>
          )}
          {collapsed && (
            <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
              <span className="text-white text-xs font-bold">A</span>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 overflow-y-auto">
          {filteredSections.map((section, si) => (
            <div key={si} className={si > 0 ? 'mt-5' : ''}>
              {/* Section label */}
              {!collapsed && (
                <p className="px-5 mb-1.5 text-[10px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-widest">
                  {section.label}
                </p>
              )}
              {collapsed && si > 0 && (
                <div className="mx-auto w-6 border-t border-gray-200 dark:border-gray-800/60 mb-3" />
              )}

              <div className="space-y-0.5 px-2.5">
                {section.items.map((item) => {
                  const active = isActive(item);
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      title={collapsed ? item.label : undefined}
                      className={`group relative flex items-center gap-3 rounded-lg transition-all duration-150 ${
                        collapsed ? 'justify-center px-0 py-2.5' : 'px-2.5 py-2'
                      } ${
                        active
                          ? 'bg-blue-500/10 text-blue-400'
                          : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.04]'
                      }`}
                    >
                      {/* Active indicator bar */}
                      {active && (
                        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 bg-blue-500 rounded-r-full" />
                      )}
                      <item.icon size={18} strokeWidth={active ? 2.2 : 1.8} className="shrink-0" />
                      {!collapsed && (
                        <span className={`text-[13px] ${active ? 'font-semibold' : 'font-medium'}`}>
                          {item.label}
                        </span>
                      )}

                      {/* Tooltip on collapsed */}
                      {collapsed && (
                        <span className="absolute left-full ml-2 px-2 py-1 rounded-md bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-xs text-gray-900 dark:text-white whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 shadow-lg">
                          {item.label}
                        </span>
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* User + collapse */}
        <div className="border-t border-gray-200 dark:border-gray-800/60">
          {/* User info */}
          <div className={`flex items-center ${collapsed ? 'justify-center py-3' : 'px-4 py-3 gap-3'}`}>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center text-white text-xs font-bold shrink-0 relative group">
              {user?.username?.[0]?.toUpperCase() || '?'}
              {collapsed && (
                <span className="absolute left-full ml-2 px-2 py-1 rounded-md bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-xs text-gray-900 dark:text-white whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 shadow-lg">
                  {user?.username}
                </span>
              )}
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium text-gray-900 dark:text-white truncate">{user?.username}</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500">{user?.is_admin ? 'Admin' : 'User'}</p>
              </div>
            )}
            {!collapsed && (
              <div className="flex items-center gap-1">
                <button
                  onClick={toggleTheme}
                  title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                  className="p-1.5 rounded-md text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.04] transition-colors"
                >
                  {theme === 'dark' ? <Sun size={16} strokeWidth={1.8} /> : <Moon size={16} strokeWidth={1.8} />}
                </button>
                <button
                  onClick={logout}
                  title="Logout"
                  className="p-1.5 rounded-md text-gray-400 dark:text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <LogOut size={16} strokeWidth={1.8} />
                </button>
              </div>
            )}
          </div>

          {collapsed && (
            <div className="flex justify-center py-2">
              <button
                onClick={toggleTheme}
                title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                className="p-1.5 rounded-md text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.04] transition-colors"
              >
                {theme === 'dark' ? <Sun size={16} strokeWidth={1.8} /> : <Moon size={16} strokeWidth={1.8} />}
              </button>
            </div>
          )}

          {/* Collapse toggle */}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className={`w-full flex items-center justify-center py-2.5 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 transition-colors border-t border-gray-200 dark:border-gray-800/40 hover:bg-black/[0.02] dark:hover:bg-white/[0.02]`}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>
      </motion.aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-8xl mx-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
