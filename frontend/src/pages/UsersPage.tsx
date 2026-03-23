import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usersAPI, authAPI } from '../api/client';
import { useAuth } from '../hooks/useAuth';
import ConfirmModal from '../components/ConfirmModal';
import { Users, Plus, Edit, Trash2, Shield, ShieldOff, UserCheck, UserX, AlertCircle, Lock, Key } from 'lucide-react';

const PROTECTED_USERNAME = 'sarbesh.tiwari@ethara.ai';

const ALL_PAGES = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'games', label: 'Games' },
  { id: 'upload', label: 'Upload' },
  { id: 'requests', label: 'Requests' },
  { id: 'users', label: 'Users' },
];

export default function UsersPage() {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newUser, setNewUser] = useState({ username: '', password: '', email: '', is_admin: false, allowed_pages: [] });
  const [error, setError] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editPages, setEditPages] = useState([]);

  const fetchUsers = () => {
    setLoading(true);
    usersAPI.list()
      .then((res) => setUsers(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchUsers(); }, []);

  const toggleNewPage = (pageId) => {
    setNewUser(u => ({
      ...u,
      allowed_pages: u.allowed_pages.includes(pageId)
        ? u.allowed_pages.filter(p => p !== pageId)
        : [...u.allowed_pages, pageId]
    }));
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const payload = { ...newUser };
      // Admins get all pages
      if (payload.is_admin) {
        payload.allowed_pages = ALL_PAGES.map(p => p.id);
      }
      await authAPI.register(payload);
      setShowCreate(false);
      setNewUser({ username: '', password: '', email: '', is_admin: false, allowed_pages: [] });
      fetchUsers();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create user');
    }
  };

  const handleToggleActive = async (userId, isActive) => {
    try {
      await usersAPI.update(userId, { is_active: !isActive });
      fetchUsers();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to update user');
    }
  };

  const handleToggleAdmin = async (userId, isAdmin) => {
    try {
      const updates: any = { is_admin: !isAdmin };
      if (!isAdmin) {
        // Promoting to admin: give all pages
        updates.allowed_pages = ALL_PAGES.map(p => p.id);
      }
      await usersAPI.update(userId, updates);
      fetchUsers();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to update user');
    }
  };

  const handleDelete = async (userId, username) => {
    if (!confirm(`Delete user "${username}"?`)) return;
    try {
      await usersAPI.delete(userId);
      fetchUsers();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to delete user');
    }
  };

  const startEditPages = (u) => {
    setEditingId(u.id);
    setEditPages(u.allowed_pages || []);
  };

  const saveEditPages = async (userId) => {
    try {
      await usersAPI.update(userId, { allowed_pages: editPages });
      setEditingId(null);
      fetchUsers();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to update user');
    }
  };

  const toggleEditPage = (pageId) => {
    setEditPages(prev =>
      prev.includes(pageId) ? prev.filter(p => p !== pageId) : [...prev, pageId]
    );
  };

  // Protected user password change
  const [showPasswordChange, setShowPasswordChange] = useState(false);
  const [secretCode, setSecretCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [pwError, setPwError] = useState('');
  const [pwSuccess, setPwSuccess] = useState('');

  const handleProtectedPasswordChange = async (e) => {
    e.preventDefault();
    setPwError('');
    setPwSuccess('');
    try {
      await usersAPI.changeProtectedPassword(secretCode, newPassword);
      setPwSuccess('Password changed successfully');
      setSecretCode('');
      setNewPassword('');
      setTimeout(() => { setShowPasswordChange(false); setPwSuccess(''); }, 2000);
    } catch (err) {
      setPwError(err.response?.data?.detail || 'Failed to change password');
    }
  };

  const isProtected = (u) => u.username === PROTECTED_USERNAME;

  // Confirmation modal
  const [confirmModal, setConfirmModal] = useState({ open: false, title: '', message: '', variant: 'danger', confirmText: '', onConfirm: null });
  const openConfirm = (opts) => setConfirmModal({ open: true, ...opts });
  const closeConfirm = () => setConfirmModal(m => ({ ...m, open: false }));

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Users</h1>
          <p className="text-gray-400 mt-1">Manage admin panel users</p>
        </div>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
        >
          <Plus size={16} /> Add User
        </button>
      </div>

      {/* Create Form */}
      <AnimatePresence>
      {showCreate && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
        >
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4">Create New User</h2>
          {error && (
            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-4">
              <AlertCircle size={14} className="text-red-400" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
          <form onSubmit={handleCreate} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Username *</label>
                <input
                  value={newUser.username}
                  onChange={(e) => setNewUser(u => ({ ...u, username: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                  required
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Password *</label>
                <input
                  type="password"
                  value={newUser.password}
                  onChange={(e) => setNewUser(u => ({ ...u, password: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                  required
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Email</label>
                <input
                  type="email"
                  value={newUser.email}
                  onChange={(e) => setNewUser(u => ({ ...u, email: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                />
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={newUser.is_admin}
                    onChange={(e) => setNewUser(u => ({ ...u, is_admin: e.target.checked }))}
                    className="rounded border-gray-600"
                  />
                  <span className="text-sm text-gray-300">Admin privileges</span>
                </label>
              </div>
            </div>

            {/* Page Access (only show when not admin) */}
            {!newUser.is_admin && (
              <div>
                <label className="block text-sm text-gray-400 mb-2">Page Access</label>
                <div className="flex flex-wrap gap-2">
                  {ALL_PAGES.map((page) => (
                    <button
                      key={page.id}
                      type="button"
                      onClick={() => toggleNewPage(page.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                        newUser.allowed_pages.includes(page.id)
                          ? 'bg-blue-500/20 border-blue-500/40 text-blue-400'
                          : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {page.label}
                    </button>
                  ))}
                </div>
                {newUser.allowed_pages.length === 0 && (
                  <p className="text-xs text-yellow-500 mt-1">No pages selected -- user won't be able to access any admin page</p>
                )}
              </div>
            )}

            <div className="flex gap-3">
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
              >
                Create User
              </button>
              <button
                type="button"
                onClick={() => { setShowCreate(false); setError(''); }}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
        </motion.div>
      )}
      </AnimatePresence>

      {/* Users Table */}
      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-800/50">
                <th className="text-left px-6 py-3 text-gray-400 font-medium">User</th>
                <th className="text-left px-6 py-3 text-gray-400 font-medium">Email</th>
                <th className="text-center px-6 py-3 text-gray-400 font-medium">Role</th>
                <th className="text-left px-6 py-3 text-gray-400 font-medium">Page Access</th>
                <th className="text-center px-6 py-3 text-gray-400 font-medium">Status</th>
                <th className="text-right px-6 py-3 text-gray-400 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-blue-600/30 flex items-center justify-center text-blue-400 text-sm font-bold">
                        {u.username[0].toUpperCase()}
                      </div>
                      <span className="text-white font-medium">{u.username}</span>
                      {u.id === currentUser?.id && (
                        <span className="text-xs text-gray-500">(you)</span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-400">{u.email || '-'}</td>
                  <td className="px-6 py-4 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      u.is_admin ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-700 text-gray-300'
                    }`}>
                      {u.is_admin ? 'Admin' : 'User'}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    {editingId === u.id ? (
                      <div className="flex flex-wrap gap-1.5 items-center">
                        {ALL_PAGES.map((page) => (
                          <button
                            key={page.id}
                            onClick={() => toggleEditPage(page.id)}
                            className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
                              editPages.includes(page.id)
                                ? 'bg-blue-500/20 border-blue-500/40 text-blue-400'
                                : 'bg-gray-800 border-gray-700 text-gray-500'
                            }`}
                          >
                            {page.label}
                          </button>
                        ))}
                        <button
                          onClick={() => saveEditPages(u.id)}
                          className="px-2 py-0.5 rounded text-[11px] font-medium bg-green-600 text-white ml-1"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => setEditingId(null)}
                          className="px-2 py-0.5 rounded text-[11px] font-medium bg-gray-700 text-gray-300"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-wrap gap-1 items-center">
                        {u.is_admin ? (
                          <span className="text-[11px] text-purple-400">All pages</span>
                        ) : (u.allowed_pages && u.allowed_pages.length > 0) ? (
                          u.allowed_pages.map((p) => (
                            <span key={p} className="px-1.5 py-0.5 rounded bg-gray-800 text-[11px] text-gray-400">{p}</span>
                          ))
                        ) : (
                          <span className="text-[11px] text-red-400">No access</span>
                        )}
                        {!u.is_admin && (
                          <button
                            onClick={() => startEditPages(u)}
                            className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-blue-400 ml-1"
                            title="Edit page access"
                          >
                            <Edit size={12} />
                          </button>
                        )}
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      u.is_active ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {u.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    {isProtected(u) ? (
                      <div className="flex items-center justify-end gap-2">
                        <span className="flex items-center gap-1 text-[11px] text-yellow-400 bg-yellow-500/10 px-2 py-1 rounded">
                          <Lock size={10} /> Protected
                        </span>
                        {currentUser?.username === PROTECTED_USERNAME && (
                          <button
                            onClick={() => setShowPasswordChange(true)}
                            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-blue-400"
                            title="Change password (requires secret code)"
                          >
                            <Key size={16} />
                          </button>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => openConfirm({
                            title: u.is_admin ? `Remove admin from "${u.username}"?` : `Make "${u.username}" admin?`,
                            message: u.is_admin
                              ? 'This user will lose admin privileges and may lose access to restricted pages.'
                              : 'This user will get full admin access to all pages.',
                            variant: 'warning',
                            confirmText: u.is_admin ? 'Remove Admin' : 'Make Admin',
                            onConfirm: () => { closeConfirm(); handleToggleAdmin(u.id, u.is_admin); },
                          })}
                          className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-purple-400"
                          title={u.is_admin ? 'Remove admin' : 'Make admin'}
                        >
                          {u.is_admin ? <ShieldOff size={16} /> : <Shield size={16} />}
                        </button>
                        <button
                          onClick={() => openConfirm({
                            title: u.is_active ? `Deactivate "${u.username}"?` : `Activate "${u.username}"?`,
                            message: u.is_active
                              ? 'This user will be locked out of the admin panel immediately.'
                              : 'This user will regain access to the admin panel.',
                            variant: u.is_active ? 'warning' : 'success',
                            confirmText: u.is_active ? 'Deactivate' : 'Activate',
                            onConfirm: () => { closeConfirm(); handleToggleActive(u.id, u.is_active); },
                          })}
                          className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-yellow-400"
                          title={u.is_active ? 'Deactivate' : 'Activate'}
                        >
                          {u.is_active ? <UserX size={16} /> : <UserCheck size={16} />}
                        </button>
                        {u.id !== currentUser?.id && (
                          <button
                            onClick={() => openConfirm({
                              title: `Delete "${u.username}"?`,
                              message: 'This will permanently remove the user account. This cannot be undone.',
                              variant: 'danger',
                              confirmText: 'Delete User',
                              onConfirm: () => { closeConfirm(); handleDelete(u.id, u.username); },
                            })}
                            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-red-400"
                            title="Delete user"
                          >
                            <Trash2 size={16} />
                          </button>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Protected user password change modal */}
      {showPasswordChange && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            className="bg-gray-900 border border-gray-800 rounded-xl p-6 w-full max-w-sm"
          >
            <h3 className="text-lg font-bold text-white mb-1">Change Password</h3>
            <p className="text-xs text-gray-500 mb-4">Protected account -- secret code required</p>
            {pwError && (
              <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-2 mb-3">
                <AlertCircle size={12} className="text-red-400" />
                <p className="text-xs text-red-400">{pwError}</p>
              </div>
            )}
            {pwSuccess && (
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-2 mb-3">
                <p className="text-xs text-green-400">{pwSuccess}</p>
              </div>
            )}
            <form onSubmit={handleProtectedPasswordChange} className="space-y-3">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Secret Code</label>
                <input
                  type="password"
                  value={secretCode}
                  onChange={(e) => setSecretCode(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                  required
                  autoFocus
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">New Password</label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
                  required
                  minLength={6}
                />
              </div>
              <div className="flex gap-2 pt-1">
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
                >
                  Change Password
                </button>
                <button
                  type="button"
                  onClick={() => { setShowPasswordChange(false); setPwError(''); setPwSuccess(''); setSecretCode(''); setNewPassword(''); }}
                  className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm"
                >
                  Cancel
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}

      <ConfirmModal
        open={confirmModal.open}
        onClose={closeConfirm}
        onConfirm={confirmModal.onConfirm}
        title={confirmModal.title}
        message={confirmModal.message}
        confirmText={confirmModal.confirmText}
        variant={confirmModal.variant}
      />
    </div>
  );
}
