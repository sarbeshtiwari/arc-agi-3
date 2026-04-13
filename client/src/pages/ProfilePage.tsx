import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { usersAPI } from '../api/client';
import { Lock, CheckCircle2, AlertCircle, Eye, EyeOff, User, Shield } from 'lucide-react';

const ROLE_LABELS = {
  super_admin: 'Super Admin',
  pl: 'Project Lead',
  ql: 'Quality Lead',
  tasker: 'Tasker',
};

const ROLE_COLORS = {
  super_admin: 'text-red-400 bg-red-500/10 border-red-500/20',
  pl: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  ql: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  tasker: 'text-green-400 bg-green-500/10 border-green-500/20',
};

export default function ProfilePage() {
  const { user } = useAuth();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');

  const userRole = user?.role || (user?.is_admin ? 'super_admin' : 'tasker');

  const handleChangePassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (newPassword.length < 6) {
      setError('New password must be at least 6 characters');
      return;
    }
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    try {
      await usersAPI.changeMyPassword(currentPassword, newPassword);
      setSuccess('Password changed successfully');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to change password');
    }
    setLoading(false);
  };

  return (
    <div className="max-w-2xl">
      <div className="mb-8">
        <h1 className="text-xl font-bold text-gray-900 dark:text-white">Profile</h1>
        <p className="text-sm text-gray-500 mt-0.5">Manage your account settings</p>
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6 mb-6">
        <div className="flex items-center gap-4 mb-6">
          <div className={`w-14 h-14 rounded-xl flex items-center justify-center text-white text-xl font-bold ${
            userRole === 'super_admin' ? 'bg-red-500' :
            userRole === 'pl' ? 'bg-purple-500' :
            userRole === 'ql' ? 'bg-blue-500' : 'bg-green-500'
          }`}>
            {(user?.display_name || user?.username)?.[0]?.toUpperCase() || '?'}
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{user?.display_name || user?.username}</h2>
            {user?.display_name && (
              <p className="text-xs text-gray-500 mt-0.5">{user.username}</p>
            )}
            <div className="flex items-center gap-2 mt-1">
              <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium border ${ROLE_COLORS[userRole]}`}>
                <Shield size={10} />
                {ROLE_LABELS[userRole]}
              </span>
              {user?.email && <span className="text-xs text-gray-500">{user.email}</span>}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          {user?.display_name && (
            <div>
              <p className="text-gray-500 text-xs mb-1">Name</p>
              <p className="text-gray-900 dark:text-white font-medium">{user.display_name}</p>
            </div>
          )}
          <div>
            <p className="text-gray-500 text-xs mb-1">Username</p>
            <p className="text-gray-900 dark:text-white font-medium flex items-center gap-1.5">
              <User size={14} className="text-gray-400" />
              {user?.username}
            </p>
          </div>
          <div>
            <p className="text-gray-500 text-xs mb-1">Role</p>
            <p className="text-gray-900 dark:text-white font-medium">{ROLE_LABELS[userRole]}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs mb-1">Member Since</p>
            <p className="text-gray-900 dark:text-white font-medium">
              {user?.created_at ? new Date(user.created_at).toLocaleDateString('en-IN') : '—'}
            </p>
          </div>
          <div>
            <p className="text-gray-500 text-xs mb-1">Status</p>
            <p className="text-green-400 font-medium text-sm flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-green-400" /> Active
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Lock size={16} /> Change Password
        </h3>

        {success && (
          <div className="mb-4 flex items-center gap-2 px-4 py-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm">
            <CheckCircle2 size={16} /> {success}
          </div>
        )}
        {error && (
          <div className="mb-4 flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
            <AlertCircle size={16} /> {error}
          </div>
        )}

        <form onSubmit={handleChangePassword} className="space-y-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">Current Password</label>
            <div className="relative">
              <input
                type={showCurrent ? 'text' : 'password'}
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
                required
              />
              <button
                type="button"
                onClick={() => setShowCurrent(!showCurrent)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
              >
                {showCurrent ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-500 mb-1.5">New Password</label>
            <div className="relative">
              <input
                type={showNew ? 'text' : 'password'}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
                minLength={6}
                required
              />
              <button
                type="button"
                onClick={() => setShowNew(!showNew)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
              >
                {showNew ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-500 mb-1.5">Confirm New Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:outline-none focus:border-blue-500/50"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading || !currentPassword || !newPassword || !confirmPassword}
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Changing...' : 'Change Password'}
          </button>
        </form>
      </div>
    </div>
  );
}
