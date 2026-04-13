import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { teamsAPI, authAPI } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users, UserPlus, UserMinus, ChevronDown, ChevronRight,
  CheckCircle2, AlertCircle, Plus, Eye, EyeOff, X, Check
} from 'lucide-react';

const ROLE_LABELS: Record<string, string> = {
  super_admin: 'Super Admin',
  pl: 'Project Lead',
  ql: 'Quality Lead',
  tasker: 'Tasker',
};

const ROLE_COLORS: Record<string, string> = {
  super_admin: 'text-red-400 bg-red-500/10',
  pl: 'text-purple-400 bg-purple-500/10',
  ql: 'text-blue-400 bg-blue-500/10',
  tasker: 'text-green-400 bg-green-500/10',
};

// ────────────────────────────────────────────────
// Multi-select Lead Picker (checkbox dropdown)
// ────────────────────────────────────────────────
function MultiLeadPicker({
  availableLeads,
  selectedIds,
  onChange,
  label,
  roleLabel,
}: {
  availableLeads: any[];
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  label: string;
  roleLabel: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const toggle = (id: string) => {
    onChange(
      selectedIds.includes(id)
        ? selectedIds.filter(x => x !== id)
        : [...selectedIds, id]
    );
  };

  const selectedNames = availableLeads
    .filter(l => selectedIds.includes(l.id))
    .map(l => l.display_name || l.username);

  return (
    <div ref={ref} className="relative">
      <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
        {label}
      </label>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-left focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50 outline-none"
      >
        <span className={selectedNames.length > 0 ? 'text-gray-900 dark:text-white' : 'text-gray-400'}>
          {selectedNames.length > 0 ? selectedNames.join(', ') : `Select ${roleLabel}(s)...`}
        </span>
        <ChevronDown size={14} className={`text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute left-0 top-full mt-1 w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl z-30 max-h-48 overflow-y-auto">
          {availableLeads.length === 0 ? (
            <div className="px-3 py-3 text-xs text-gray-500 text-center">
              No {roleLabel}s available
            </div>
          ) : (
            availableLeads.map(lead => {
              const selected = selectedIds.includes(lead.id);
              return (
                <button
                  type="button"
                  key={lead.id}
                  onClick={() => toggle(lead.id)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2.5 text-left hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors ${
                    selected ? 'bg-blue-500/5' : ''
                  }`}
                >
                  <div className={`w-4 h-4 rounded flex items-center justify-center border ${
                    selected
                      ? 'bg-blue-500 border-blue-500 text-white'
                      : 'border-gray-300 dark:border-gray-600'
                  }`}>
                    {selected && <Check size={10} />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {lead.display_name || lead.username}
                    </p>
                    <p className="text-[10px] text-gray-500 truncate">{lead.username}</p>
                  </div>
                </button>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────
// Create User Modal
// ────────────────────────────────────────────────
function CreateUserModal({
  open, onClose, onSuccess, isSuperAdmin, isPL, isQL, currentUserId, allUsers,
}: {
  open: boolean;
  onClose: () => void;
  onSuccess: (msg: string) => void;
  isSuperAdmin: boolean;
  isPL: boolean;
  isQL: boolean;
  currentUserId: string;
  allUsers: any[];
}) {
  const [form, setForm] = useState({
    emailPrefix: '',
    displayName: '',
    password: '',
    confirmPassword: '',
    role: 'tasker' as string,
    teamLeadIds: [] as string[],
  });
  const [error, setError] = useState('');
  const [creating, setCreating] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const backdropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setForm({ emailPrefix: '', displayName: '', password: '', confirmPassword: '', role: 'tasker', teamLeadIds: [] });
      setError('');
      setShowPassword(false);
    }
  }, [open]);

  // Available leads based on selected role
  const availableQLs = allUsers.filter(u => u.role === 'ql');
  const availablePLs = allUsers.filter(u => u.role === 'pl');

  const needsLeadPicker = (isSuperAdmin || isPL) && (form.role === 'tasker' || form.role === 'ql');
  const leadOptions = form.role === 'tasker' ? availableQLs : availablePLs;
  const leadRoleLabel = form.role === 'tasker' ? 'QL' : 'PL';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const username = `${form.emailPrefix}@ethara.ai`;

    if (!form.emailPrefix || form.emailPrefix.length < 2) {
      setError('Username prefix must be at least 2 characters');
      return;
    }
    if (/[^a-zA-Z0-9._-]/.test(form.emailPrefix)) {
      setError('Username prefix can only contain letters, numbers, dots, hyphens, underscores');
      return;
    }
    if (!form.displayName.trim()) {
      setError('Name is required');
      return;
    }
    if (!form.password || form.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setCreating(true);
    try {
      const payload: any = {
        username,
        password: form.password,
        display_name: form.displayName.trim(),
        email: username,
      };

      if (isSuperAdmin) {
        payload.role = form.role;
        if (needsLeadPicker && form.teamLeadIds.length > 0) {
          payload.team_lead_ids = form.teamLeadIds;
        }
      } else if (isPL) {
        payload.role = form.role;
        if (form.role === 'tasker' && form.teamLeadIds.length > 0) {
          payload.team_lead_ids = form.teamLeadIds;
        } else if (form.role === 'ql') {
          payload.team_lead_ids = [currentUserId];
        }
      } else if (isQL) {
        payload.role = 'tasker';
        payload.team_lead_ids = [currentUserId];
      }

      await authAPI.register(payload);
      onSuccess(`User "${form.displayName.trim()}" (${username}) created successfully`);
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create user');
    }
    setCreating(false);
  };

  if (!open) return null;

  return (
    <div
      ref={backdropRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={e => { if (e.target === backdropRef.current) onClose(); }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        transition={{ duration: 0.15 }}
        className="w-full max-w-lg mx-4 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-2xl shadow-2xl"
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800">
          <h3 className="text-base font-semibold text-gray-900 dark:text-white">
            {isQL ? 'Create New Tasker' : isPL ? 'Create Team Member' : 'Create New User'}
          </h3>
          <button onClick={onClose} className="p-1 text-gray-400 hover:text-gray-200 rounded-md transition-colors">
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="px-6 py-5">
          {error && (
            <div className="mb-4 flex items-center gap-2 px-3 py-2.5 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
              <AlertCircle size={14} /> {error}
            </div>
          )}

          <div className="space-y-4">
            {/* Row 1: Username (email@ethara.ai) + Name */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
                  Username *
                </label>
                <div className="flex">
                  <input
                    type="text"
                    value={form.emailPrefix}
                    onChange={e => setForm(f => ({ ...f, emailPrefix: e.target.value.toLowerCase() }))}
                    className="flex-1 min-w-0 px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-l-lg text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50 outline-none"
                    placeholder="john.doe"
                    autoFocus
                    required
                  />
                  <span className="inline-flex items-center px-3 py-2.5 bg-gray-100 dark:bg-gray-700 border border-l-0 border-gray-200 dark:border-gray-700 rounded-r-lg text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
                    @ethara.ai
                  </span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
                  Name *
                </label>
                <input
                  type="text"
                  value={form.displayName}
                  onChange={e => setForm(f => ({ ...f, displayName: e.target.value }))}
                  className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50 outline-none"
                  placeholder="John Doe"
                  required
                />
              </div>
            </div>

            {/* Row 2: Password + Confirm Password */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
                  Password *
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={form.password}
                    onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                    className="w-full px-3 py-2.5 pr-10 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50 outline-none"
                    placeholder="min 6 characters"
                    required
                    minLength={6}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-300"
                  >
                    {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
                  Confirm Password *
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={form.confirmPassword}
                    onChange={e => setForm(f => ({ ...f, confirmPassword: e.target.value }))}
                    className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50 outline-none"
                    placeholder="repeat password"
                    required
                    minLength={6}
                  />
                  {form.confirmPassword && form.password === form.confirmPassword && (
                    <CheckCircle2 size={14} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-green-400" />
                  )}
                  {form.confirmPassword && form.password !== form.confirmPassword && (
                    <AlertCircle size={14} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-red-400" />
                  )}
                </div>
              </div>
            </div>

            {/* Row 3: Role (super admin only) */}
            {(isSuperAdmin || isPL) && (
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
                  Role *
                </label>
                <select
                  value={form.role}
                  onChange={e => setForm(f => ({ ...f, role: e.target.value, teamLeadIds: [] }))}
                  className="w-full px-3 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500/50 outline-none"
                >
                  <option value="tasker">Tasker</option>
                  <option value="ql">Quality Lead (QL)</option>
                  {isSuperAdmin && <option value="pl">Project Lead (PL)</option>}
                  {isSuperAdmin && <option value="super_admin">Super Admin</option>}
                </select>
              </div>
            )}

            {/* Row 4: Lead picker (conditional) */}
            {needsLeadPicker && (
              <MultiLeadPicker
                availableLeads={leadOptions}
                selectedIds={form.teamLeadIds}
                onChange={(ids) => setForm(f => ({ ...f, teamLeadIds: ids }))}
                label={form.role === 'tasker' ? 'Reporting QL(s)' : 'Reporting PL(s)'}
                roleLabel={leadRoleLabel}
              />
            )}
          </div>

          <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-gray-200 dark:border-gray-800">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-500 hover:text-gray-300 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={creating}
              className="flex items-center gap-2 px-5 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              {creating ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Plus size={16} />
              )}
              {isQL ? 'Create Tasker' : isPL ? 'Create Member' : 'Create User'}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}

// ────────────────────────────────────────────────
// Assign-to-Lead Picker (for Super Admin unassigned list)
// ────────────────────────────────────────────────
function LeadPicker({
  userRole,
  allUsers,
  onSelect,
  disabled,
}: {
  userRole: string;
  allUsers: any[];
  onSelect: (leadId: string) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // tasker → pick a QL, ql → pick a PL
  const leadRole = userRole === 'tasker' ? 'ql' : userRole === 'ql' ? 'pl' : null;
  const leads = leadRole ? allUsers.filter(u => u.role === leadRole) : [];

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  if (!leadRole || leads.length === 0) {
    return <span className="text-[10px] text-gray-600 italic">No {leadRole === 'ql' ? 'QLs' : 'PLs'} to assign to</span>;
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        disabled={disabled}
        className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-colors disabled:opacity-50"
      >
        <UserPlus size={12} />
        Add {leadRole === 'ql' ? 'QL' : 'PL'}
        <ChevronDown size={10} className={`ml-0.5 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 w-52 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl z-40">
          <div className="px-3 py-2 border-b border-gray-200 dark:border-gray-700 text-[10px] text-gray-500 font-medium uppercase tracking-wider">
            Select {ROLE_LABELS[leadRole]}
          </div>
          {leads.map(lead => (
            <button
              key={lead.id}
              onClick={() => { onSelect(lead.id); setOpen(false); }}
              className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors"
            >
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-[10px] font-bold ${
                leadRole === 'ql' ? 'bg-blue-500' : 'bg-purple-500'
              }`}>
                {(lead.display_name || lead.username)[0].toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-gray-900 dark:text-white truncate">
                  {lead.display_name || lead.username}
                </p>
                <p className="text-[10px] text-gray-500 truncate">{lead.username}</p>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────
// Main Page
// ────────────────────────────────────────────────
export default function TeamManagementPage() {
  const { user, isQL, isPL, isSuperAdmin } = useAuth();
  const navigate = useNavigate();
  const [team, setTeam] = useState<any[]>([]);
  const [plTeam, setPlTeam] = useState<{ qls: any[]; taskers: any[] }>({ qls: [], taskers: [] });
  const [unassigned, setUnassigned] = useState<any[]>([]);
  const [allUsers, setAllUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [assigning, setAssigning] = useState<string | null>(null);
  const [showUnassigned, setShowUnassigned] = useState(false);
  const [success, setSuccess] = useState('');
  const [showCreateUser, setShowCreateUser] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      if (isSuperAdmin) {
        const [allRes, unRes] = await Promise.all([
          teamsAPI.allUsers(),
          teamsAPI.unassigned(),
        ]);
        setAllUsers(allRes.data);
        setUnassigned(unRes.data);
      } else if (isPL) {
        const [teamRes, unRes] = await Promise.all([
          teamsAPI.myTeam(),
          teamsAPI.unassigned(),
        ]);
        setPlTeam(teamRes.data);
        setUnassigned(unRes.data);
        const allPLUsers = [
          ...(teamRes.data.qls || []),
          ...(teamRes.data.taskers || []),
          ...unRes.data,
        ];
        setAllUsers(allPLUsers);
      } else if (isQL) {
        const [teamRes, unRes] = await Promise.all([
          teamsAPI.myTeam(),
          teamsAPI.unassigned(),
        ]);
        setTeam(teamRes.data);
        setUnassigned(unRes.data);
      }
    } catch {}
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, []);

  const handleAssign = async (userId: string, leadId: string) => {
    setAssigning(userId);
    try {
      await teamsAPI.assign(userId, leadId);
      setSuccess('User assigned successfully');
      setTimeout(() => setSuccess(''), 3000);
      fetchData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to assign user');
    }
    setAssigning(null);
  };

  const handleUnassign = async (userId: string, leadId: string) => {
    setAssigning(userId);
    try {
      await teamsAPI.unassign(userId, leadId);
      setSuccess('User unassigned from lead');
      setTimeout(() => setSuccess(''), 3000);
      fetchData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to unassign');
    }
    setAssigning(null);
  };

  const handleCreateSuccess = (msg: string) => {
    setSuccess(msg);
    setTimeout(() => setSuccess(''), 3000);
    fetchData();
  };

  if (loading) {
    return (
      <div className="py-16 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">Loading team data...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">Team Management</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {isSuperAdmin ? 'Manage all user assignments' : isQL ? 'Manage your taskers' : isPL ? 'Your QLs and their taskers' : 'View your team'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(isSuperAdmin || isQL || isPL) && (
            <button
              onClick={() => setShowCreateUser(true)}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Plus size={16} />
              {isQL ? 'Create Tasker' : isPL ? 'Create Member' : 'Create User'}
            </button>
          )}
          {unassigned.length > 0 && (isQL || isSuperAdmin || isPL) && (
            <button
              onClick={() => setShowUnassigned(!showUnassigned)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <UserPlus size={16} />
              Assign Users ({unassigned.length})
            </button>
          )}
        </div>
      </div>

      {/* Create User Modal */}
      <AnimatePresence>
        {showCreateUser && (
          <CreateUserModal
            open={showCreateUser}
            onClose={() => setShowCreateUser(false)}
            onSuccess={handleCreateSuccess}
            isSuperAdmin={isSuperAdmin}
            isPL={isPL}
            isQL={isQL}
            currentUserId={user.id}
            allUsers={allUsers}
          />
        )}
      </AnimatePresence>

      {/* Success toast */}
      <AnimatePresence>
        {success && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mb-4 flex items-center gap-2 px-4 py-3 bg-green-500/10 border border-green-500/20 rounded-xl text-green-400 text-sm"
          >
            <CheckCircle2 size={16} /> {success}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Unassigned Users Panel */}
      {showUnassigned && unassigned.length > 0 && (
        <div className="mb-6 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              <AlertCircle size={14} className="text-yellow-400" />
              Unassigned Users
            </h3>
            <button onClick={() => setShowUnassigned(false)} className="text-xs text-gray-500 hover:text-gray-300">Close</button>
          </div>
          <div className="divide-y divide-gray-100 dark:divide-gray-800/50">
            {unassigned.map((u: any) => (
              <div key={u.id} className="flex items-center gap-3 px-4 py-3">
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-white text-xs font-bold">
                  {(u.display_name || u.username)[0].toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {u.display_name || u.username}
                  </p>
                  <p className="text-[10px] text-gray-500">{u.username}</p>
                  <span className={`inline-flex px-2 py-0.5 rounded-full text-[10px] font-medium ${ROLE_COLORS[u.role] || ''}`}>
                    {ROLE_LABELS[u.role] || u.role}
                  </span>
                </div>
                {/* QL: simple "Add to My Team" for taskers */}
                {isQL && u.role === 'tasker' && (
                  <button
                    onClick={() => handleAssign(u.id, user.id)}
                    disabled={assigning === u.id}
                    className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-colors disabled:opacity-50"
                  >
                    <UserPlus size={12} /> Add to My Team
                  </button>
                )}
                {isPL && (
                  <LeadPicker
                    userRole={u.role}
                    allUsers={[...plTeam.qls, { id: user.id, role: 'pl', username: user.username, display_name: user.display_name }]}
                    onSelect={(leadId) => handleAssign(u.id, leadId)}
                    disabled={assigning === u.id}
                  />
                )}
                {isSuperAdmin && (
                  <LeadPicker
                    userRole={u.role}
                    allUsers={allUsers}
                    onSelect={(leadId) => handleAssign(u.id, leadId)}
                    disabled={assigning === u.id}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main team view */}
      {isSuperAdmin ? (
        <SuperAdminTeamView
          users={allUsers}
          onUnassign={handleUnassign}
          onAssign={handleAssign}
          assigning={assigning}
          navigate={navigate}
        />
      ) : isPL ? (
        <PLTeamView qls={plTeam.qls} taskers={plTeam.taskers} navigate={navigate} />
      ) : (
        <TeamMembersList
          members={team}
          onUnassign={isQL ? (id: string) => handleUnassign(id, user.id) : null}
          assigning={assigning}
          navigate={navigate}
         />
      )}
    </div>
  );
}

function PLTeamView({ qls, taskers, navigate }: { qls: any[]; taskers: any[]; navigate: (path: string) => void }) {
  const [expandedQL, setExpandedQL] = useState<string | null>(null);

  const getQLTaskers = (qlId: string) =>
    taskers.filter((t: any) => t.team_lead_ids && t.team_lead_ids.includes(qlId));

  if (qls.length === 0) {
    return (
      <div className="py-16 text-center bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
        <Users size={32} className="mx-auto text-gray-600 mb-3" />
        <p className="text-gray-500 text-sm">No QLs assigned to you yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
        Your Quality Leads
        <span className="px-2 py-0.5 rounded-full text-[10px] font-medium text-blue-400 bg-blue-500/10">
          {qls.length}
        </span>
      </h3>
      {qls.map((ql: any) => {
        const qlTaskers = getQLTaskers(ql.id);
        const isExpanded = expandedQL === ql.id;

        return (
          <div key={ql.id} className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
            <div
              className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors rounded-t-xl"
              onClick={() => setExpandedQL(isExpanded ? null : ql.id)}
            >
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-bold">
                {(ql.display_name || ql.username)[0].toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {ql.display_name || ql.username}
                </p>
                <p className="text-xs text-gray-500">
                  {ql.username} · {qlTaskers.length} tasker{qlTaskers.length !== 1 ? 's' : ''}
                </p>
              </div>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium text-blue-400 bg-blue-500/10">
                Quality Lead
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); navigate(`/dashboard/team/${ql.id}`); }}
                className="px-3 py-1.5 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 rounded-lg text-[10px] font-medium transition-colors"
              >
                View Details
              </button>
              {qlTaskers.length > 0 && (
                isExpanded
                  ? <ChevronDown size={14} className="text-gray-400" />
                  : <ChevronRight size={14} className="text-gray-400" />
              )}
            </div>
            {isExpanded && qlTaskers.length > 0 && (
              <div className="border-t border-gray-100 dark:border-gray-800/50">
                {qlTaskers.map((t: any) => (
                  <div key={t.id} className="flex items-center gap-3 px-4 py-2 pl-14 hover:bg-gray-50 dark:hover:bg-white/[0.02] cursor-pointer"
                    onClick={() => navigate(`/dashboard/team/tasker/${t.id}`)}
                  >
                    <div className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center text-white text-[10px] font-bold">
                      {(t.display_name || t.username)[0].toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-300">
                        {t.display_name || t.username}
                      </p>
                      <p className="text-[10px] text-gray-500">{t.username}</p>
                    </div>
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-medium text-green-400 bg-green-500/10">
                      Tasker
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {taskers.length > 0 && (
        <p className="text-xs text-gray-500 mt-4">
          Total taskers across your QLs: {taskers.length}
        </p>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────
// QL / PL Team List
// ────────────────────────────────────────────────
function TeamMembersList({ members, onUnassign, assigning, navigate }: { members: any[]; onUnassign: ((id: string) => void) | null; assigning: string | null; navigate: (path: string) => void }) {
  if (members.length === 0) {
    return (
      <div className="py-16 text-center bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
        <Users size={32} className="mx-auto text-gray-600 mb-3" />
        <p className="text-gray-500 text-sm">No team members yet</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-800">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
          Team Members ({members.length})
        </h3>
      </div>
      <div className="divide-y divide-gray-100 dark:divide-gray-800/50">
        {members.map((m: any) => (
          <div key={m.id} className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors cursor-pointer"
            onClick={() => navigate(`/dashboard/team/tasker/${m.id}`)}
          >
            <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-xs font-bold">
              {(m.display_name || m.username)[0].toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {m.display_name || m.username}
              </p>
              <p className="text-xs text-gray-500">{m.username}</p>
            </div>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${ROLE_COLORS[m.role] || ''}`}>
              {ROLE_LABELS[m.role] || m.role}
            </span>
            {m.team_lead_usernames && m.team_lead_usernames.length > 0 && (
              <span className="text-[10px] text-gray-500">
                → {m.team_lead_usernames.join(', ')}
              </span>
            )}
            {onUnassign && (
              <button
                onClick={() => onUnassign(m.id)}
                disabled={assigning === m.id}
                className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-colors disabled:opacity-50"
                title="Remove from team"
              >
                <UserMinus size={14} />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────
// Super Admin: Full Org View with reassignment
// ────────────────────────────────────────────────
function SuperAdminTeamView({
  users, onUnassign, onAssign, assigning, navigate,
}: {
  users: any[];
  onUnassign: (userId: string, leadId: string) => void;
  onAssign: (userId: string, leadId: string) => void;
  assigning: string | null;
  navigate: (path: string) => void;
}) {
  const [expandedLead, setExpandedLead] = useState<string | null>(null);

  const pls = users.filter((u: any) => u.role === 'pl');
  const qls = users.filter((u: any) => u.role === 'ql');
  const taskers = users.filter((u: any) => u.role === 'tasker');
  const admins = users.filter((u: any) => u.role === 'super_admin');

  // Use junction table data: find all users who have this lead in their team_lead_ids
  const getTeamMembers = (leadId: string) =>
    users.filter((u: any) => u.team_lead_ids && u.team_lead_ids.includes(leadId));

  const renderUserGroup = (title: string, groupUsers: any[], color: string) => (
    <div className="mb-6">
      <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
        {title}
        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${color}`}>{groupUsers.length}</span>
      </h3>
      {groupUsers.length === 0 ? (
        <p className="text-gray-500 text-xs py-3">None</p>
      ) : (
        <div className="space-y-2">
          {groupUsers.map((u: any) => {
            const teamMembers = getTeamMembers(u.id);
            const isExpanded = expandedLead === u.id;

            return (
              <div key={u.id} className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
                <div
                  className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors rounded-t-xl"
                  onClick={() => setExpandedLead(isExpanded ? null : u.id)}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold ${
                    u.role === 'pl' ? 'bg-purple-500' : u.role === 'ql' ? 'bg-blue-500' : 'bg-red-500'
                  }`}>
                    {(u.display_name || u.username)[0].toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {u.display_name || u.username}
                    </p>
                    <p className="text-xs text-gray-500">
                      {u.username}
                      {u.team_lead_usernames && u.team_lead_usernames.length > 0
                        ? ` · Reports to: ${u.team_lead_usernames.join(', ')}`
                        : (u.role === 'tasker' || u.role === 'ql') ? ' · No lead assigned' : ''}
                      {teamMembers.length > 0 ? ` · ${teamMembers.length} member${teamMembers.length > 1 ? 's' : ''}` : ''}
                    </p>
                  </div>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${ROLE_COLORS[u.role] || ''}`}>
                    {ROLE_LABELS[u.role] || u.role}
                  </span>
                  {(u.role === 'ql' || u.role === 'pl') && (
                    <button
                      onClick={(e) => { e.stopPropagation(); navigate(`/dashboard/team/${u.id}`); }}
                      className="px-3 py-1.5 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 rounded-lg text-[10px] font-medium transition-colors"
                    >
                      View Details
                    </button>
                  )}
                  {u.role === 'tasker' && (
                    <button
                      onClick={(e) => { e.stopPropagation(); navigate(`/dashboard/team/tasker/${u.id}`); }}
                      className="px-3 py-1.5 bg-green-600/10 hover:bg-green-600/20 text-green-400 rounded-lg text-[10px] font-medium transition-colors"
                    >
                      View Details
                    </button>
                  )}
                  {u.role !== 'super_admin' && u.role !== 'pl' && (
                    <div onClick={e => e.stopPropagation()}>
                      <LeadPicker
                        userRole={u.role}
                        allUsers={users}
                        onSelect={(leadId) => onAssign(u.id, leadId)}
                        disabled={assigning === u.id}
                      />
                    </div>
                  )}
                  {teamMembers.length > 0 && (
                    isExpanded ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />
                  )}
                </div>
                {isExpanded && teamMembers.length > 0 && (
                  <div className="border-t border-gray-100 dark:border-gray-800/50">
                    {teamMembers.map((m: any) => (
                      <div key={m.id} className="flex items-center gap-3 px-4 py-2 pl-14 hover:bg-gray-50 dark:hover:bg-white/[0.02] cursor-pointer"
                        onClick={() => {
                          if (m.role === 'tasker') navigate(`/dashboard/team/tasker/${m.id}`);
                          else if (m.role === 'ql' || m.role === 'pl') navigate(`/dashboard/team/${m.id}`);
                        }}
                      >
                        <div className="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center text-white text-[10px] font-bold">
                          {(m.display_name || m.username)[0].toUpperCase()}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-gray-300">
                            {m.display_name || m.username}
                          </p>
                          <p className="text-[10px] text-gray-500">{m.username}</p>
                        </div>
                        <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${ROLE_COLORS[m.role] || ''}`}>
                          {ROLE_LABELS[m.role] || m.role}
                        </span>
                        <button
                          onClick={(e) => { e.stopPropagation(); onUnassign(m.id, u.id); }}
                          disabled={assigning === m.id}
                          className="p-1 text-gray-500 hover:text-red-400 transition-colors disabled:opacity-50"
                          title="Remove from this lead"
                        >
                          <UserMinus size={12} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  return (
    <div>
      {renderUserGroup('Super Admins', admins, ROLE_COLORS.super_admin)}
      {renderUserGroup('Project Leads', pls, ROLE_COLORS.pl)}
      {renderUserGroup('Quality Leads', qls, ROLE_COLORS.ql)}
      {renderUserGroup('Taskers', taskers, ROLE_COLORS.tasker)}
    </div>
  );
}
