import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { teamsAPI, gamesAPI } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Gamepad2, CheckCircle2, XCircle, Clock,
  FileText, Calendar, ChevronDown, ChevronRight, Filter,
  Trophy, Play, AlertTriangle, Download, History,
  Eye, RefreshCw
} from 'lucide-react';

const STATUS_CONFIG: Record<string, { label: string; color: string; bg: string; icon: any }> = {
  draft: { label: 'Draft', color: 'text-gray-400', bg: 'bg-gray-500/10', icon: FileText },
  pending_ql: { label: 'Pending QL', color: 'text-amber-400', bg: 'bg-amber-500/10', icon: Clock },
  ql_approved: { label: 'QL Approved', color: 'text-blue-400', bg: 'bg-blue-500/10', icon: CheckCircle2 },
  pending_pl: { label: 'Pending PL', color: 'text-purple-400', bg: 'bg-purple-500/10', icon: Clock },
  approved: { label: 'Approved', color: 'text-green-400', bg: 'bg-green-500/10', icon: CheckCircle2 },
  rejected: { label: 'Rejected', color: 'text-red-400', bg: 'bg-red-500/10', icon: XCircle },
};

const AUDIT_ACTION_LABELS: Record<string, { label: string; color: string }> = {
  game_uploaded: { label: 'Uploaded', color: 'bg-blue-500' },
  game_updated: { label: 'Updated', color: 'bg-cyan-500' },
  game_submitted: { label: 'Submitted for Review', color: 'bg-amber-500' },
  game_resubmitted: { label: 'Resubmitted', color: 'bg-amber-500' },
  game_approved_by_ql: { label: 'Approved by QL', color: 'bg-blue-500' },
  game_approved_by_pl: { label: 'Approved by PL', color: 'bg-green-500' },
  game_approved_by_admin: { label: 'Approved by Admin', color: 'bg-green-500' },
  game_rejected_by_ql: { label: 'Rejected by QL', color: 'bg-red-500' },
  game_rejected_by_pl: { label: 'Rejected by PL', color: 'bg-red-500' },
  game_file_replaced: { label: 'Files Replaced', color: 'bg-cyan-500' },
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

export default function TaskerDetailPage() {
  const { userId } = useParams<{ userId: string }>();
  const navigate = useNavigate();

  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [expandedGame, setExpandedGame] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  const fetchData = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const res = await teamsAPI.taskerDetail(userId, fromDate, toDate);
      setData(res.data);
      setError('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load tasker details');
    }
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, [userId]);

  const handleDownload = async (gameId: string) => {
    setDownloading(gameId);
    try {
      const res = await gamesAPI.download(gameId);
      const blob = new Blob([res.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${gameId}.zip`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch {
      alert('Download failed');
    }
    setDownloading(null);
  };

  if (loading) {
    return (
      <div className="py-16 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">Loading tasker details...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-16 text-center">
        <AlertTriangle size={32} className="mx-auto text-red-400 mb-3" />
        <p className="text-red-400 text-sm">{error || 'Tasker not found'}</p>
        <button onClick={() => navigate(-1)} className="mt-4 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg text-sm hover:bg-gray-700">
          Go Back
        </button>
      </div>
    );
  }

  const { user: member, games, stats } = data;

  const filteredGames = statusFilter === 'all'
    ? games
    : games.filter((g: any) => g.approval_status === statusFilter);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={() => navigate(-1)}
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
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-3 mb-6">
        {[
          { label: 'Total', value: stats.total, color: 'text-white', bg: 'bg-gray-800' },
          { label: 'Draft', value: stats.draft, color: 'text-gray-400', bg: 'bg-gray-800/50' },
          { label: 'Pending QL', value: stats.pending_ql, color: 'text-amber-400', bg: 'bg-amber-500/5' },
          { label: 'QL Approved', value: stats.ql_approved || 0, color: 'text-blue-400', bg: 'bg-blue-500/5' },
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
          onClick={fetchData}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-colors"
        >
          Apply
        </button>
        {(fromDate || toDate) && (
          <button
            onClick={() => { setFromDate(''); setToDate(''); setTimeout(fetchData, 0); }}
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

      {/* Games List */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
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
          filteredGames.map((g: any) => {
            const sc = STATUS_CONFIG[g.approval_status] || STATUS_CONFIG.draft;
            const StatusIcon = sc.icon;
            const isExpanded = expandedGame === g.id;

            return (
              <div key={g.id} className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl">
                {/* Game Header */}
                <div
                  className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedGame(isExpanded ? null : g.id)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">{g.name || g.game_id}</p>
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${sc.color} ${sc.bg}`}>
                        <StatusIcon size={10} />
                        {sc.label}
                      </span>
                    </div>
                    <p className="text-[10px] text-gray-500 font-mono">{g.game_id}</p>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-gray-500">
                    {g.gameplay?.play_count > 0 && (
                      <span className="flex items-center gap-1 text-cyan-400" title="Times played by uploader">
                        <Gamepad2 size={12} /> {g.gameplay.play_count}×
                      </span>
                    )}
                    {g.gameplay?.levels_cleared > 0 && (
                      <span className="flex items-center gap-1 text-green-400" title="Levels cleared">
                        <CheckCircle2 size={12} /> {g.gameplay.levels_cleared}
                      </span>
                    )}
                    {g.gameplay?.total_game_overs > 0 && (
                      <span className="flex items-center gap-1 text-red-400" title="Game overs">
                        <XCircle size={12} /> {g.gameplay.total_game_overs}
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      <History size={12} />
                      {g.update_count} update{g.update_count !== 1 ? 's' : ''}
                    </span>
                    <span className="flex items-center gap-1">
                      <Play size={12} /> {g.total_plays}
                    </span>
                    <span className="flex items-center gap-1 text-green-400">
                      <Trophy size={12} /> {g.total_wins}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => { e.stopPropagation(); window.open(`/play/${g.game_id}`, '_blank'); }}
                      className="p-1.5 text-blue-400 hover:bg-blue-500/10 rounded-lg transition-colors"
                      title="Play"
                    >
                      <Play size={14} />
                    </button>
                    {g.has_game_file && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDownload(g.game_id); }}
                        disabled={downloading === g.game_id}
                        className="p-1.5 text-gray-400 hover:text-green-400 hover:bg-green-500/10 rounded-lg transition-colors disabled:opacity-50"
                        title="Download Files"
                      >
                        {downloading === g.game_id
                          ? <RefreshCw size={14} className="animate-spin" />
                          : <Download size={14} />
                        }
                      </button>
                    )}
                    {isExpanded ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
                  </div>
                </div>

                {/* Expanded: Game Details + History Timeline */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden border-t border-gray-100 dark:border-gray-800/50"
                    >
                      <div className="px-4 py-4 space-y-4">
                        {/* Game Info Grid */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                          <InfoCell label="Version" value={g.version || '1'} />
                          <InfoCell label="Game Code" value={g.game_code} mono />
                          <InfoCell label="FPS" value={String(g.default_fps)} />
                          <InfoCell label="Active" value={g.is_active ? 'Yes' : 'No'} color={g.is_active ? 'text-green-400' : 'text-red-400'} />
                          <InfoCell label="Created" value={formatDateTime(g.created_at)} />
                          <InfoCell label="Last Updated" value={formatDateTime(g.updated_at)} />
                          <InfoCell label="First Activity" value={formatDateTime(g.first_update)} />
                          <InfoCell label="Last Activity" value={formatDateTime(g.last_update)} />
                        </div>

                        {/* Gameplay Stats */}
                        {g.gameplay && g.gameplay.play_count > 0 && (
                          <div className="p-3 bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700/50 rounded-xl">
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
                              <Gamepad2 size={12} className="text-cyan-400" />
                              Uploader's Gameplay Stats
                            </p>
                            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-3">
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Times Played</p>
                                <p className="text-sm font-bold text-cyan-400">{g.gameplay.play_count}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Levels Cleared</p>
                                <p className="text-sm font-bold text-green-400">{g.gameplay.levels_cleared}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Max Level</p>
                                <p className="text-sm font-bold text-blue-400">{g.gameplay.max_level_reached}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Game Overs</p>
                                <p className="text-sm font-bold text-red-400">{g.gameplay.total_game_overs}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Wins</p>
                                <p className="text-sm font-bold text-green-400">{g.gameplay.wins}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Losses</p>
                                <p className="text-sm font-bold text-red-400">{g.gameplay.losses}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Total Time</p>
                                <p className="text-sm font-bold text-amber-400">{formatTime(g.gameplay.total_time_secs)}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Avg Time/Session</p>
                                <p className="text-sm font-bold text-amber-400">{formatTime(g.gameplay.avg_time_secs)}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Total Actions</p>
                                <p className="text-sm font-bold text-gray-300">{g.gameplay.total_actions}</p>
                              </div>
                              <div>
                                <p className="text-[10px] text-gray-500 mb-0.5">Last Played</p>
                                <p className="text-xs font-medium text-gray-300">{formatDateTime(g.gameplay.last_played_at)}</p>
                              </div>
                            </div>
                          </div>
                        )}

                        {g.description && (
                          <div>
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Description</p>
                            <p className="text-xs text-gray-300">{g.description}</p>
                          </div>
                        )}

                        {/* Assignments */}
                        <div className="flex flex-wrap gap-4 text-xs">
                          {g.ql_username && (
                            <span className="text-gray-500">QL: <span className="text-blue-400">{g.ql_username}</span></span>
                          )}
                          {g.pl_username && (
                            <span className="text-gray-500">PL: <span className="text-purple-400">{g.pl_username}</span></span>
                          )}
                        </div>

                        {g.rejection_reason && (
                          <div className="px-3 py-2 bg-red-500/5 border border-red-500/20 rounded-lg">
                            <p className="text-[10px] text-red-400/70 uppercase tracking-wider mb-0.5">Rejection Reason</p>
                            <p className="text-xs text-red-400">{g.rejection_reason}</p>
                          </div>
                        )}

                        {/* Files */}
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-gray-500 uppercase tracking-wider">Files:</span>
                          <span className={`text-xs ${g.has_game_file ? 'text-green-400' : 'text-gray-600'}`}>
                            {g.has_game_file ? '✓ game.py' : '✗ game.py'}
                          </span>
                          <span className={`text-xs ${g.has_metadata_file ? 'text-green-400' : 'text-gray-600'}`}>
                            {g.has_metadata_file ? '✓ metadata.json' : '✗ metadata.json'}
                          </span>
                        </div>

                        {/* History Timeline */}
                        {g.audit_log && g.audit_log.length > 0 && (
                          <div>
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                              <History size={12} />
                              History Timeline ({g.audit_log.length} events)
                            </p>
                            <div className="relative pl-5">
                              <div className="absolute left-[7px] top-1 bottom-1 w-px bg-gray-700" />
                              {g.audit_log.map((entry: any, idx: number) => {
                                const actionConfig = AUDIT_ACTION_LABELS[entry.action] || { label: entry.action, color: 'bg-gray-500' };
                                return (
                                  <div key={entry.id} className="relative mb-3 last:mb-0">
                                    <div className={`absolute -left-5 top-1.5 w-3 h-3 rounded-full ${actionConfig.color} border-2 border-gray-900`} />
                                    <div className="ml-1">
                                      <div className="flex items-center gap-2 flex-wrap">
                                        <span className="text-xs font-medium text-gray-200">{actionConfig.label}</span>
                                        <span className="text-[10px] text-gray-500">by {entry.username}</span>
                                        <span className="text-[10px] text-gray-600">{formatDateTime(entry.created_at)}</span>
                                      </div>
                                      {entry.details && (
                                         <div className="mt-0.5 text-[10px] text-gray-500">
                                          {entry.details.reason && <span>Reason: {entry.details.reason}</span>}
                                          {entry.details.message && <span>Note: {entry.details.message}</span>}
                                          {entry.details.previous_status && <span>From: {entry.details.previous_status}</span>}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {(!g.audit_log || g.audit_log.length === 0) && (
                          <p className="text-xs text-gray-600 italic">No history recorded for this game</p>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function InfoCell({ label, value, mono, color }: { label: string; value: string; mono?: boolean; color?: string }) {
  return (
    <div>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">{label}</p>
      <p className={`text-xs ${color || 'text-gray-300'} ${mono ? 'font-mono' : ''}`}>{value}</p>
    </div>
  );
}
