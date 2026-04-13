import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { approvalAPI, gamesAPI } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Gamepad2, CheckCircle2, XCircle, Clock, ChevronDown, ChevronUp,
  Play, Download, AlertCircle, History, Send, Shield, MessageSquare,
  Square, CheckSquare, Package, BookOpen, Upload, RefreshCw, X
} from 'lucide-react';

const APPROVAL_STATUS_LABELS = {
  draft: 'Draft',
  pending_ql: 'Pending QL Review',
  ql_approved: 'QL Approved',
  pending_pl: 'Pending PL Review',
  approved: 'Approved',
  rejected: 'Rejected',
};

const STATUS_STYLES = {
  draft: { color: 'text-gray-400', bg: 'bg-gray-500/10 border-gray-500/20', icon: AlertCircle },
  pending_ql: { color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20', icon: Clock },
  ql_approved: { color: 'text-blue-400', bg: 'bg-blue-500/10 border-blue-500/20', icon: CheckCircle2 },
  pending_pl: { color: 'text-purple-400', bg: 'bg-purple-500/10 border-purple-500/20', icon: Clock },
  approved: { color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/20', icon: CheckCircle2 },
  rejected: { color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20', icon: XCircle },
};

function AuditTimeline({ gameId }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    approvalAPI.getAuditLog(gameId)
      .then(res => setEntries(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [gameId]);

  if (loading) return <div className="py-4 text-center text-gray-500 text-xs">Loading history...</div>;
  if (entries.length === 0) return <div className="py-4 text-center text-gray-500 text-xs">No history yet</div>;

  return (
    <div className="space-y-3">
      {entries.map((entry, i) => (
        <div key={entry.id} className="flex gap-3">
          <div className="flex flex-col items-center">
            <div className="w-2 h-2 rounded-full bg-blue-400 mt-1.5" />
            {i < entries.length - 1 && <div className="flex-1 w-px bg-gray-700/50 mt-1" />}
          </div>
          <div className="pb-3">
            <p className="text-xs text-gray-300">
              <span className="font-medium">{entry.username || 'System'}</span>
              {' — '}
              <span className="text-gray-500">{entry.action.replace(/_/g, ' ')}</span>
            </p>
            {entry.details?.reason && (
              <p className="text-xs text-gray-500 mt-0.5 italic">"{entry.details.reason}"</p>
            )}
            {entry.details?.message && (
              <p className="text-xs text-gray-500 mt-0.5 italic">Remarks: "{entry.details.message}"</p>
            )}
            <p className="text-[10px] text-gray-600 mt-0.5">
              {new Date(entry.created_at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ReviewQueuePage() {
  const { user, isQL, isPL, isSuperAdmin } = useAuth();
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);
  const [actionLoading, setActionLoading] = useState(null);
  const [filter, setFilter] = useState('pending');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkDownloading, setBulkDownloading] = useState(false);
  const [confirmModal, setConfirmModal] = useState<{
    type: 'approve' | 'reject';
    game: any;
  } | null>(null);
  const [confirmComment, setConfirmComment] = useState('');
  const [updateModal, setUpdateModal] = useState<{ gameId: string; gameName: string } | null>(null);
  const [updateFiles, setUpdateFiles] = useState<{ game_file: File | null; metadata_file: File | null }>({ game_file: null, metadata_file: null });
  const [updating, setUpdating] = useState(false);
  const navigate = useNavigate();

  const fetchGames = () => {
    setLoading(true);
    const apiCall = isSuperAdmin
      ? approvalAPI.allGames()
      : isPL
        ? approvalAPI.plQueue()
        : approvalAPI.qlQueue();

    apiCall
      .then(res => setGames(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchGames(); }, []);

  const getPendingStatus = () => {
    if (isSuperAdmin) return ['pending_ql', 'pending_pl'];
    if (isPL) return ['pending_pl'];
    return ['pending_ql'];
  };

  const filtered = (() => {
    if (filter === 'pending') return games.filter(g => getPendingStatus().includes(g.approval_status));
    if (filter === 'approved') return games.filter(g => g.approval_status === 'approved' || g.approval_status === 'ql_approved');
    if (filter === 'rejected') return games.filter(g => g.approval_status === 'rejected');
    return games;
  })();

  const handleApprove = async () => {
    if (!confirmModal || confirmModal.type !== 'approve') return;
    const game = confirmModal.game;
    setActionLoading(game.id);
    try {
      if (isSuperAdmin) {
        await approvalAPI.adminApprove(game.id, confirmComment || null);
      } else if (isQL) {
        await approvalAPI.qlReview(game.id, 'approve', confirmComment || null);
      } else if (isPL) {
        await approvalAPI.plReview(game.id, 'approve', confirmComment || null);
      }
      setConfirmModal(null);
      setConfirmComment('');
      fetchGames();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to approve');
    }
    setActionLoading(null);
  };

  const handleReject = async () => {
    if (!confirmModal || confirmModal.type !== 'reject') return;
    if (!confirmComment.trim()) {
      alert('Please provide a rejection reason');
      return;
    }
    const game = confirmModal.game;
    setActionLoading(game.id);
    try {
      if (isQL) {
        await approvalAPI.qlReview(game.id, 'reject', confirmComment);
      } else if (isPL) {
        await approvalAPI.plReview(game.id, 'reject', confirmComment);
      }
      setConfirmModal(null);
      setConfirmComment('');
      fetchGames();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to reject');
    }
    setActionLoading(null);
  };

  const canReview = (game) => {
    if (isSuperAdmin) return ['pending_ql', 'pending_pl', 'ql_approved'].includes(game.approval_status);
    if (isQL) return game.approval_status === 'pending_ql';
    if (isPL) return game.approval_status === 'pending_pl';
    return false;
  };

  const pendingCount = games.filter(g => getPendingStatus().includes(g.approval_status)).length;

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const selectAllFiltered = () => {
    const filteredIds = filtered.map(g => g.id);
    const allSelected = filteredIds.every(id => selectedIds.has(id));
    if (allSelected) {
      setSelectedIds(prev => {
        const next = new Set(prev);
        filteredIds.forEach(id => next.delete(id));
        return next;
      });
    } else {
      setSelectedIds(prev => {
        const next = new Set(prev);
        filteredIds.forEach(id => next.add(id));
        return next;
      });
    }
  };

  const handleBulkDownload = async (ids?: string[]) => {
    const downloadIds = ids || Array.from(selectedIds);
    if (downloadIds.length === 0) return;
    setBulkDownloading(true);
    try {
      const gameIds = downloadIds.map(id => {
        const game = games.find(g => g.id === id);
        return game?.game_id || id;
      });
      const res = await gamesAPI.bulkDownload(gameIds);
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = `games-${gameIds.length}.zip`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Download failed');
    }
    setBulkDownloading(false);
  };

  const handleDownloadByStatus = (status: string) => {
    let targets: any[];
    if (status === 'all') targets = games;
    else if (status === 'pending') targets = games.filter(g => getPendingStatus().includes(g.approval_status));
    else if (status === 'approved') targets = games.filter(g => ['approved', 'ql_approved'].includes(g.approval_status));
    else targets = games.filter(g => g.approval_status === status);
    if (targets.length === 0) return;
    handleBulkDownload(targets.map(g => g.id));
  };

  const handleUpdateFiles = async () => {
    if (!updateModal || (!updateFiles.game_file && !updateFiles.metadata_file)) return;
    setUpdating(true);
    try {
      const formData = new FormData();
      if (updateFiles.game_file) formData.append('game_file', updateFiles.game_file);
      if (updateFiles.metadata_file) formData.append('metadata_file', updateFiles.metadata_file);
      await gamesAPI.updateFiles(updateModal.gameId, formData);
      setUpdateModal(null);
      setUpdateFiles({ game_file: null, metadata_file: null });
      fetchGames();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Update failed');
    }
    setUpdating(false);
  };

  const filterOptions = [
    { value: 'pending', label: 'Needs Review', count: pendingCount },
    { value: 'approved', label: 'Approved', count: games.filter(g => ['approved', 'ql_approved'].includes(g.approval_status)).length },
    { value: 'rejected', label: 'Rejected', count: games.filter(g => g.approval_status === 'rejected').length },
    { value: 'all', label: 'All', count: games.length },
  ];

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-bold text-gray-900 dark:text-white">
          {isSuperAdmin ? 'All Games Overview' : 'Review Queue'}
        </h1>
        <p className="text-sm text-gray-500 mt-0.5">
          {pendingCount} game{pendingCount !== 1 ? 's' : ''} pending review
        </p>
      </div>

      <div className="flex gap-2 mb-6 overflow-x-auto pb-1">
        {filterOptions.map(opt => (
          <button
            key={opt.value}
            onClick={() => setFilter(opt.value)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
              filter === opt.value
                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                : 'text-gray-500 hover:text-gray-300 border border-transparent'
            }`}
          >
            {opt.label}
            {opt.count > 0 && <span className="ml-1.5 text-[10px] opacity-60">({opt.count})</span>}
          </button>
        ))}
      </div>

      {games.length > 0 && (
        <div className="flex items-center gap-3 mb-4 p-3 bg-gray-900/50 border border-gray-800 rounded-xl">
          <button
            onClick={selectAllFiltered}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors"
          >
            {filtered.length > 0 && filtered.every(g => selectedIds.has(g.id))
              ? <CheckSquare size={16} className="text-blue-400" />
              : <Square size={16} />
            }
            {filtered.length > 0 && filtered.every(g => selectedIds.has(g.id)) ? 'Deselect All' : 'Select All'}
          </button>

          {selectedIds.size > 0 && (
            <>
              <span className="text-xs text-gray-500">|</span>
              <span className="text-xs text-blue-400">{selectedIds.size} selected</span>
              <button
                onClick={() => handleBulkDownload()}
                disabled={bulkDownloading}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-colors disabled:opacity-50"
              >
                {bulkDownloading
                  ? <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white" />
                  : <Package size={12} />
                }
                Download Selected
              </button>
            </>
          )}

          <div className="flex-1" />
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-gray-600">Quick:</span>
            {[
              { key: 'all', label: 'All' },
              { key: 'approved', label: 'Approved' },
              { key: 'rejected', label: 'Rejected' },
              { key: 'pending', label: 'Pending' },
            ].map(opt => (
              <button
                key={opt.key}
                onClick={() => handleDownloadByStatus(opt.key)}
                disabled={bulkDownloading}
                className="px-2 py-1 text-[10px] text-gray-500 hover:text-gray-300 hover:bg-gray-800 rounded transition-colors disabled:opacity-50"
              >
                <Download size={10} className="inline mr-0.5" />{opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading ? (
        <div className="py-16 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3" />
          <p className="text-gray-500 text-sm">Loading queue...</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="py-16 text-center">
          <CheckCircle2 size={32} className="mx-auto text-green-400 mb-3" />
          <p className="text-gray-500 text-sm">
            {filter === 'pending' ? 'No games pending your review!' : 'No games in this category.'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map(game => {
            const status = STATUS_STYLES[game.approval_status] || STATUS_STYLES.draft;
            const StatusIcon = status.icon;
            const isExpanded = expandedId === game.id;
            const reviewable = canReview(game);

            return (
              <div key={game.id} className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
                <div
                  className="flex items-center gap-4 px-5 py-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : game.id)}
                >
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleSelect(game.id); }}
                    className="shrink-0 text-gray-500 hover:text-blue-400 transition-colors"
                  >
                    {selectedIds.has(game.id)
                      ? <CheckSquare size={18} className="text-blue-400" />
                      : <Square size={18} />
                    }
                  </button>
                  <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-gray-800 flex items-center justify-center shrink-0">
                    <Gamepad2 size={18} className="text-gray-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">{game.game_id}</p>
                    <p className="text-xs text-gray-500 truncate">
                      by {game.uploader_username || 'Unknown'} • v{game.version || 1}
                    </p>
                  </div>
                  <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border ${status.bg} ${status.color}`}>
                    <StatusIcon size={12} />
                    {APPROVAL_STATUS_LABELS[game.approval_status]}
                  </span>
                  {reviewable && (
                    <span className="px-2 py-0.5 bg-yellow-500/10 text-yellow-400 text-[10px] font-medium rounded-full border border-yellow-500/20">
                      ACTION NEEDED
                    </span>
                  )}
                  {isExpanded ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
                </div>

                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="px-5 pb-4 border-t border-gray-100 dark:border-gray-800/50 pt-4">
                        <div className="grid sm:grid-cols-2 gap-4 mb-4">
                          <div className="space-y-2 text-xs">
                            <div className="flex justify-between">
                              <span className="text-gray-500">Uploaded By</span>
                              <span className="text-gray-300">{game.uploader_username || '—'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Game Code</span>
                              <span className="text-gray-300 font-mono">{game.game_code || '—'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">QL</span>
                              <span className="text-gray-300">{game.ql_username || 'Unassigned'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">PL</span>
                              <span className="text-gray-300">{game.pl_username || 'Unassigned'}</span>
                            </div>
                            {game.description && (
                              <div className="mt-2">
                                <p className="text-gray-500 mb-1">Description</p>
                                <p className="text-gray-400 text-xs">{game.description}</p>
                              </div>
                            )}
                            {game.game_rules && (
                              <div className="mt-2">
                                <p className="text-gray-500 mb-1 flex items-center gap-1"><BookOpen size={11} /> Game Rules</p>
                                <p className="text-gray-400 text-xs whitespace-pre-wrap">{game.game_rules}</p>
                              </div>
                            )}
                            {game.rejection_reason && (
                              <div className="mt-2 p-2 bg-red-500/5 border border-red-500/10 rounded-lg">
                                <p className="text-red-400 text-xs font-medium">Last Rejection:</p>
                                <p className="text-red-300/80 text-xs mt-0.5">{game.rejection_reason}</p>
                              </div>
                            )}
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-2 flex items-center gap-1"><History size={12} /> History</p>
                            <AuditTimeline gameId={game.id} />
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-gray-100 dark:border-gray-800/50">
                          {game.game_code && (
                            <button
                              onClick={(e) => { e.stopPropagation(); window.open(`/play/${game.game_id}`, '_blank'); }}
                              className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-xs font-medium transition-colors"
                            >
                              <Play size={12} /> Test Play
                            </button>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              window.open(`/api/games/${game.game_id}/download`, '_blank');
                            }}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-xs font-medium transition-colors"
                          >
                            <Download size={12} /> Download
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); setUpdateModal({ gameId: game.game_id, gameName: game.name || game.game_id }); setUpdateFiles({ game_file: null, metadata_file: null }); }}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-600/80 hover:bg-orange-500 text-white rounded-lg text-xs font-medium transition-colors"
                          >
                            <Upload size={12} /> Update Files
                          </button>

                          {reviewable && (
                            <>
                              <div className="flex-1" />
                              <button
                                onClick={(e) => { e.stopPropagation(); setConfirmComment(''); setConfirmModal({ type: 'approve', game }); }}
                                className="flex items-center gap-1.5 px-4 py-1.5 bg-green-600 hover:bg-green-500 text-white rounded-lg text-xs font-medium transition-colors"
                              >
                                {isSuperAdmin ? <Shield size={12} /> : <CheckCircle2 size={12} />}
                                {isSuperAdmin ? 'Admin Approve' : 'Approve'}
                              </button>
                              {!isSuperAdmin && (
                                <button
                                  onClick={(e) => { e.stopPropagation(); setConfirmComment(''); setConfirmModal({ type: 'reject', game }); }}
                                  className="flex items-center gap-1.5 px-4 py-1.5 bg-red-600 hover:bg-red-500 text-white rounded-lg text-xs font-medium transition-colors"
                                >
                                  <XCircle size={12} /> Reject
                                </button>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      )}

      <AnimatePresence>
        {confirmModal && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={() => setConfirmModal(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-md p-6"
            >
              {confirmModal.type === 'approve' ? (
                <>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-full bg-green-500/10 flex items-center justify-center">
                      <CheckCircle2 size={20} className="text-green-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Confirm Approval</h3>
                      <p className="text-xs text-gray-500">This action cannot be undone</p>
                    </div>
                  </div>

                  <div className="mb-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                    <p className="text-sm text-gray-300">
                      Approve <span className="font-mono text-white">{confirmModal.game.game_id}</span>?
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      by {confirmModal.game.uploader_username || 'Unknown'} • v{confirmModal.game.version || 1}
                    </p>
                  </div>

                  <div className="mb-4">
                    <label className="flex items-center gap-1.5 text-sm text-gray-300 mb-1.5">
                      <MessageSquare size={14} />
                      Remarks <span className="text-gray-600 text-xs">(optional)</span>
                    </label>
                    <textarea
                      value={confirmComment}
                      onChange={(e) => setConfirmComment(e.target.value)}
                      placeholder="Add a comment or feedback for the tasker..."
                      rows={3}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-green-500/50 focus:ring-1 focus:ring-green-500/20 resize-none"
                      autoFocus
                    />
                  </div>

                  <div className="flex items-center gap-3 justify-end">
                    <button
                      onClick={() => setConfirmModal(null)}
                      className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleApprove}
                      disabled={actionLoading === confirmModal.game.id}
                      className="flex items-center gap-1.5 px-5 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {actionLoading === confirmModal.game.id
                        ? <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-white" />
                        : <CheckCircle2 size={14} />
                      }
                      Yes, Approve
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center">
                      <XCircle size={20} className="text-red-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Confirm Rejection</h3>
                      <p className="text-xs text-gray-500">This will send the game back to the tasker</p>
                    </div>
                  </div>

                  <div className="mb-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                    <p className="text-sm text-gray-300">
                      Reject <span className="font-mono text-white">{confirmModal.game.game_id}</span>?
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      by {confirmModal.game.uploader_username || 'Unknown'} • v{confirmModal.game.version || 1}
                    </p>
                  </div>

                  <div className="mb-4">
                    <label className="flex items-center gap-1.5 text-sm text-gray-300 mb-1.5">
                      <MessageSquare size={14} />
                      Rejection Reason <span className="text-red-400 text-xs">*</span>
                    </label>
                    <textarea
                      value={confirmComment}
                      onChange={(e) => setConfirmComment(e.target.value)}
                      placeholder="Explain why this game is being rejected..."
                      rows={3}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/20 resize-none"
                      autoFocus
                    />
                  </div>

                  <div className="flex items-center gap-3 justify-end">
                    <button
                      onClick={() => setConfirmModal(null)}
                      className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleReject}
                      disabled={actionLoading === confirmModal.game.id || !confirmComment.trim()}
                      className="flex items-center gap-1.5 px-5 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {actionLoading === confirmModal.game.id
                        ? <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-white" />
                        : <XCircle size={14} />
                      }
                      Yes, Reject
                    </button>
                  </div>
                </>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {updateModal && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={() => setUpdateModal(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-md p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-semibold text-white">Update Game Files</h3>
                <button onClick={() => setUpdateModal(null)} className="p-1 text-gray-500 hover:text-gray-300"><X size={18} /></button>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-3 mb-4 text-xs text-gray-400">
                Game: <span className="text-white font-mono">{updateModal.gameName}</span>
                <p className="mt-1 text-orange-400">
                  {isQL ? 'Updating will set status to Pending PL Review.' :
                   isPL ? 'Updating will keep the game approved.' :
                   'Updating will require re-review from the next level.'}
                </p>
              </div>

              <div className="space-y-3 mb-5">
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Game File (.py)</label>
                  <input
                    type="file"
                    accept=".py"
                    onChange={(e) => setUpdateFiles(prev => ({ ...prev, game_file: e.target.files?.[0] || null }))}
                    className="w-full text-xs text-gray-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-gray-700 file:text-gray-200 hover:file:bg-gray-600"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Metadata File (.json)</label>
                  <input
                    type="file"
                    accept=".json"
                    onChange={(e) => setUpdateFiles(prev => ({ ...prev, metadata_file: e.target.files?.[0] || null }))}
                    className="w-full text-xs text-gray-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-gray-700 file:text-gray-200 hover:file:bg-gray-600"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 justify-end">
                <button onClick={() => setUpdateModal(null)} className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors">Cancel</button>
                <button
                  onClick={handleUpdateFiles}
                  disabled={updating || (!updateFiles.game_file && !updateFiles.metadata_file)}
                  className="flex items-center gap-1.5 px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {updating ? <RefreshCw size={14} className="animate-spin" /> : <Upload size={14} />}
                  Update Files
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
