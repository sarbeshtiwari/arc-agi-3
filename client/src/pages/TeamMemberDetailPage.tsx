import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { teamsAPI } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Users, Gamepad2, CheckCircle2, XCircle, Clock,
  FileText, Calendar, ChevronDown, ChevronRight, Filter,
  Trophy, Play, AlertTriangle, Eye
} from 'lucide-react';

const STATUS_CONFIG: Record<string, { label: string; color: string; bg: string; icon: any }> = {
  draft: { label: 'Draft', color: 'text-gray-400', bg: 'bg-gray-500/10', icon: FileText },
  pending_ql: { label: 'Pending QL', color: 'text-amber-400', bg: 'bg-amber-500/10', icon: Clock },
  ql_approved: { label: 'QL Approved', color: 'text-blue-400', bg: 'bg-blue-500/10', icon: CheckCircle2 },
  pending_pl: { label: 'Pending PL', color: 'text-purple-400', bg: 'bg-purple-500/10', icon: Clock },
  approved: { label: 'Approved', color: 'text-green-400', bg: 'bg-green-500/10', icon: CheckCircle2 },
  rejected: { label: 'Rejected', color: 'text-red-400', bg: 'bg-red-500/10', icon: XCircle },
};

const ROLE_COLORS: Record<string, string> = {
  super_admin: 'bg-red-500',
  pl: 'bg-purple-500',
  ql: 'bg-blue-500',
  tasker: 'bg-green-500',
};

const ROLE_LABELS: Record<string, string> = {
  super_admin: 'Super Admin',
  pl: 'Project Lead',
  ql: 'Quality Lead',
  tasker: 'Tasker',
};

function formatDate(d: string) {
  return new Date(d).toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    timeZone: 'Asia/Kolkata',
  });
}

function formatDateTime(d: string) {
  return new Date(d).toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
    timeZone: 'Asia/Kolkata',
  });
}

function formatTime(secs: number) {
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

export default function TeamMemberDetailPage() {
  const { userId } = useParams<{ userId: string }>();
  const navigate = useNavigate();

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [expandedTasker, setExpandedTasker] = useState<string | null>(null);

  const fetchData = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const res = await teamsAPI.memberDetail(userId, fromDate, toDate);
      setData(res.data);
      setError('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load member details');
    }
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, [userId]);

  const applyDateFilter = () => { fetchData(); };

  const clearDateFilter = () => {
    setFromDate('');
    setToDate('');
    setTimeout(fetchData, 0);
  };

  if (loading) {
    return (
      <div className="py-16 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">Loading member details...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-16 text-center">
        <AlertTriangle size={32} className="mx-auto text-red-400 mb-3" />
        <p className="text-red-400 text-sm">{error || 'Member not found'}</p>
        <button onClick={() => navigate('/dashboard/team')} className="mt-4 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg text-sm hover:bg-gray-700">
          Back to Team
        </button>
      </div>
    );
  }

  const { user: member, taskers, games, stats } = data;

  const filteredGames = statusFilter === 'all'
    ? games
    : games.filter((g: any) => g.approval_status === statusFilter);

  const getTaskerGames = (taskerId: string) =>
    filteredGames.filter((g: any) => g.uploaded_by === taskerId);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={() => navigate('/dashboard/team')}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-gray-500 hover:text-gray-300"
        >
          <ArrowLeft size={20} />
        </button>
        <div className="flex items-center gap-3 flex-1">
          <div className={`w-10 h-10 rounded-full ${ROLE_COLORS[member.role] || 'bg-gray-500'} flex items-center justify-center text-white text-sm font-bold`}>
            {(member.display_name || member.username)[0].toUpperCase()}
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900 dark:text-white">
              {member.display_name || member.username}
            </h1>
            <p className="text-xs text-gray-500">
              {member.username} · {ROLE_LABELS[member.role] || member.role}
              {member.team_lead_usernames?.length > 0 && ` · Reports to: ${member.team_lead_usernames.join(', ')}`}
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        {[
          { label: 'Total', value: stats.total, color: 'text-white', bg: 'bg-gray-800' },
          { label: 'Draft', value: stats.draft, color: 'text-gray-400', bg: 'bg-gray-800/50' },
          { label: 'Pending QL', value: stats.pending_ql, color: 'text-amber-400', bg: 'bg-amber-500/5' },
          { label: 'Pending PL', value: stats.pending_pl, color: 'text-purple-400', bg: 'bg-purple-500/5' },
          { label: 'Approved', value: stats.approved, color: 'text-green-400', bg: 'bg-green-500/5' },
          { label: 'Rejected', value: stats.rejected, color: 'text-red-400', bg: 'bg-red-500/5' },
        ].map(s => (
          <div key={s.label} className={`${s.bg} border border-gray-200 dark:border-gray-800 rounded-xl p-3`}>
            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{s.label}</p>
            <p className={`text-xl font-bold ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Date Filter */}
      <div className="flex flex-wrap items-center gap-3 mb-4 p-3 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
        <Filter size={14} className="text-gray-400" />
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">From</label>
          <input
            type="date"
            value={fromDate}
            onChange={e => setFromDate(e.target.value)}
            className="px-2 py-1.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-xs text-gray-900 dark:text-white"
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">To</label>
          <input
            type="date"
            value={toDate}
            onChange={e => setToDate(e.target.value)}
            className="px-2 py-1.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-xs text-gray-900 dark:text-white"
          />
        </div>
        <button
          onClick={applyDateFilter}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-colors"
        >
          Apply
        </button>
        {(fromDate || toDate) && (
          <button
            onClick={clearDateFilter}
            className="px-3 py-1.5 bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-700 rounded-lg text-xs font-medium transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Status Filter */}
      <div className="flex flex-wrap gap-2 mb-6">
        {[
          { key: 'all', label: 'All', count: stats.total },
          { key: 'draft', label: 'Draft', count: stats.draft },
          { key: 'pending_ql', label: 'Pending QL', count: stats.pending_ql },
          { key: 'pending_pl', label: 'Pending PL', count: stats.pending_pl },
          { key: 'approved', label: 'Approved', count: stats.approved },
          { key: 'rejected', label: 'Rejected', count: stats.rejected },
        ].map(f => (
          <button
            key={f.key}
            onClick={() => setStatusFilter(f.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
              statusFilter === f.key
                ? 'bg-blue-600 text-white border-blue-600'
                : 'bg-white dark:bg-gray-900 text-gray-500 border-gray-200 dark:border-gray-800 hover:border-blue-500/30'
            }`}
          >
            {f.label} ({f.count})
          </button>
        ))}
      </div>

      {/* Taskers Section */}
      {taskers.length > 0 && (
        <div className="mb-6">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <Users size={16} className="text-blue-400" />
            Assigned Taskers
            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium text-green-400 bg-green-500/10">
              {taskers.length}
            </span>
          </h2>
          <div className="space-y-2">
            {taskers.map((t: any) => {
              const tGames = getTaskerGames(t.id);
              const isExpanded = expandedTasker === t.id;

              return (
                <div key={t.id} className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
                  <div
                    className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors"
                    onClick={() => navigate(`/dashboard/team/tasker/${t.id}`)}
                  >
                    <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-xs font-bold">
                      {(t.display_name || t.username)[0].toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {t.display_name || t.username}
                      </p>
                      <p className="text-xs text-gray-500">{t.username}</p>
                    </div>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-medium text-gray-400 bg-gray-500/10">
                      {tGames.length} game{tGames.length !== 1 ? 's' : ''}
                    </span>
                    {tGames.length > 0 && (
                      isExpanded
                        ? <ChevronDown size={14} className="text-gray-400" />
                        : <ChevronRight size={14} className="text-gray-400" />
                    )}
                  </div>
                  <AnimatePresence>
                    {isExpanded && tGames.length > 0 && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden border-t border-gray-100 dark:border-gray-800/50"
                      >
                        {tGames.map((g: any) => (
                          <GameRow key={g.id} game={g} navigate={navigate} />
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* All Games Table */}
      <div>
        <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
          <Gamepad2 size={16} className="text-cyan-400" />
          Games
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium text-cyan-400 bg-cyan-500/10">
            {filteredGames.length}
          </span>
        </h2>

        {filteredGames.length === 0 ? (
          <div className="py-12 text-center bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
            <Gamepad2 size={28} className="mx-auto text-gray-600 mb-2" />
            <p className="text-gray-500 text-sm">No games found for this filter</p>
          </div>
        ) : (
          <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-800">
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Game</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Uploader</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Played</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Levels</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Game Overs</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Time</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Last Played</th>
                    <th className="px-4 py-3 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Plays</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800/50">
                  {filteredGames.map((g: any) => {
                    const sc = STATUS_CONFIG[g.approval_status] || STATUS_CONFIG.draft;
                    const StatusIcon = sc.icon;
                    return (
                      <tr key={g.id} className="hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors">
                        <td className="px-4 py-3">
                          <p className="text-sm font-medium text-gray-900 dark:text-white">{g.name || g.game_id}</p>
                          <p className="text-[10px] text-gray-500 font-mono">{g.game_id}</p>
                        </td>
                        <td className="px-4 py-3">
                          <p className="text-xs text-gray-300">{g.uploader_display_name || g.uploader_username || '—'}</p>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${sc.color} ${sc.bg}`}>
                            <StatusIcon size={10} />
                            {sc.label}
                          </span>
                          {g.rejection_reason && (
                            <p className="text-[10px] text-red-400/70 mt-0.5 max-w-[200px] truncate" title={g.rejection_reason}>
                              {g.rejection_reason}
                            </p>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-xs font-medium ${g.gameplay?.play_count > 0 ? 'text-cyan-400' : 'text-gray-600'}`}>
                            {g.gameplay?.play_count ?? 0}×
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-xs">
                            <span className="text-green-400 font-medium">{g.gameplay?.levels_cleared ?? 0}</span>
                            {g.gameplay?.max_level_reached > 0 && (
                              <span className="text-gray-500 ml-1">/ max {g.gameplay.max_level_reached}</span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`text-xs font-medium ${g.gameplay?.total_game_overs > 0 ? 'text-red-400' : 'text-gray-600'}`}>
                            {g.gameplay?.total_game_overs ?? 0}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs text-amber-400">
                            {g.gameplay?.total_time_secs > 0 ? formatTime(g.gameplay.total_time_secs) : '—'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs text-gray-400">
                            {g.gameplay?.last_played_at ? formatDateTime(g.gameplay.last_played_at) : '—'}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2 text-xs">
                            <span className="text-gray-400 flex items-center gap-0.5">
                              <Play size={10} /> {g.total_plays}
                            </span>
                            <span className="text-green-400 flex items-center gap-0.5">
                              <Trophy size={10} /> {g.total_wins}
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function GameRow({ game: g, navigate }: { game: any; navigate: any }) {
  const sc = STATUS_CONFIG[g.approval_status] || STATUS_CONFIG.draft;
  const StatusIcon = sc.icon;

  return (
    <div className="flex items-center gap-3 px-4 py-2.5 pl-14 hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors">
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-gray-900 dark:text-gray-200">{g.name || g.game_id}</p>
        <p className="text-[10px] text-gray-500 font-mono">{g.game_id}</p>
      </div>
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${sc.color} ${sc.bg}`}>
        <StatusIcon size={10} />
        {sc.label}
      </span>
      {g.gameplay?.play_count > 0 && (
        <span className="text-[10px] text-cyan-400" title="Times played">{g.gameplay.play_count}× played</span>
      )}
      {g.gameplay?.levels_cleared > 0 && (
        <span className="text-[10px] text-green-400" title="Levels cleared">{g.gameplay.levels_cleared} cleared</span>
      )}
      {g.gameplay?.last_played_at && (
        <span className="text-[10px] text-gray-500" title="Last played">{formatDateTime(g.gameplay.last_played_at)}</span>
      )}
      {!g.gameplay?.last_played_at && (
        <span className="text-[10px] text-gray-500">{formatDate(g.created_at)}</span>
      )}
    </div>
  );
}
