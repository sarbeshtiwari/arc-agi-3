import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { requestsAPI } from '../api/client';
import ConfirmModal from '../components/ConfirmModal';
import {
  Clock, Check, X, Trash2, Code, Eye, AlertCircle,
  CheckCircle, XCircle, MessageSquare, User, Mail, ExternalLink,
  Video, FolderOpen, ChevronDown, ChevronRight, Play, Gamepad2
} from 'lucide-react';

function StatusBadge({ status }: any) {
  const map = {
    pending:  { bg: 'bg-yellow-500/20', text: 'text-yellow-400', Icon: Clock },
    approved: { bg: 'bg-green-500/20',  text: 'text-green-400',  Icon: CheckCircle },
    rejected: { bg: 'bg-red-500/20',    text: 'text-red-400',    Icon: XCircle },
  };
  const s = map[status] || map.pending;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${s.bg} ${s.text}`}>
      <s.Icon size={11} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function SourceModal({ source, onClose }: any) {
  if (!source) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-3xl max-h-[80vh] flex flex-col mx-4">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2 text-white font-semibold"><Code size={18} /> Source Code</div>
          <button onClick={onClose} className="p-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"><X size={18} /></button>
        </div>
        {source.metadata && (
          <div className="px-6 py-3 border-b border-gray-800 bg-gray-800/40">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              {source.metadata.game_id && <div><span className="text-gray-500">Game ID:</span> <span className="text-gray-200">{source.metadata.game_id}</span></div>}
              {source.metadata.default_fps != null && <div><span className="text-gray-500">FPS:</span> <span className="text-gray-200">{source.metadata.default_fps}</span></div>}
              {source.metadata.tags?.length > 0 && (
                <div className="col-span-full"><span className="text-gray-500">Tags:</span> {source.metadata.tags.map(t => <span key={t} className="inline-block px-2 py-0.5 mr-1 rounded bg-gray-700 text-gray-300 text-xs">{t}</span>)}</div>
              )}
            </div>
          </div>
        )}
        <div className="flex-1 overflow-auto p-6">
          <pre className="text-sm text-gray-300 font-mono whitespace-pre-wrap break-words leading-relaxed">{source.source_code || '(no source available)'}</pre>
        </div>
      </div>
    </div>
  );
}

export default function RequestedGamesPage() {
  const navigate = useNavigate();
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedId, setExpandedId] = useState(null);
  const [rejectingId, setRejectingId] = useState(null);
  const [rejectNote, setRejectNote] = useState('');
  const [sourceData, setSourceData] = useState(null);
  const [sourceLoading, setSourceLoading] = useState(null);
  const [actionLoading, setActionLoading] = useState({});
  const [confirmModal, setConfirmModal] = useState<any>({ open: false });
  const [playLoading, setPlayLoading] = useState(null);

  const openConfirm = (opts) => setConfirmModal({ open: true, ...opts });
  const closeConfirm = () => setConfirmModal(m => ({ ...m, open: false }));
  const toggle = (id) => setExpandedId(prev => prev === id ? null : id);

  const fetchRequests = () => {
    setLoading(true);
    setError('');
    requestsAPI.list('all')
      .then((res) => setRequests(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load requests'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchRequests(); }, []);

  const handleApprove = async (id) => {
    setActionLoading(p => ({ ...p, [id]: 'approve' }));
    try { await requestsAPI.review(id, 'approve'); fetchRequests(); }
    catch (err) { alert(err.response?.data?.detail || 'Failed to approve'); }
    finally { setActionLoading(p => { const n = { ...p }; delete n[id]; return n; }); }
  };

  const handleReject = async (id) => {
    setActionLoading(p => ({ ...p, [id]: 'reject' }));
    try { await requestsAPI.review(id, 'reject', rejectNote || null); setRejectingId(null); setRejectNote(''); fetchRequests(); }
    catch (err) { alert(err.response?.data?.detail || 'Failed to reject'); }
    finally { setActionLoading(p => { const n = { ...p }; delete n[id]; return n; }); }
  };

  const handleDelete = async (id) => {
    setActionLoading(p => ({ ...p, [id]: 'delete' }));
    try { await requestsAPI.delete(id); fetchRequests(); }
    catch (err) { alert(err.response?.data?.detail || 'Failed to delete'); }
    finally { setActionLoading(p => { const n = { ...p }; delete n[id]; return n; }); }
  };

  const handleViewSource = async (id) => {
    setSourceLoading(id);
    try { const res = await requestsAPI.getSource(id); setSourceData(res.data); }
    catch (err) { alert(err.response?.data?.detail || 'Failed to load source'); }
    finally { setSourceLoading(null); }
  };

  const handlePlay = async (req) => {
    setPlayLoading(req.id);
    try {
      const [gameRes, metaRes] = await Promise.all([
        requestsAPI.getFile(req.id, 'game'),
        requestsAPI.getFile(req.id, 'metadata'),
      ]);
      const gameFile = new File([gameRes.data], `${req.game_id}.py`, { type: 'text/x-python' });
      const metadataFile = new File([metaRes.data], 'metadata.json', { type: 'application/json' });
      navigate('/play-direct', { state: { gameFile, metadataFile, playerName: 'Admin' } });
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to load game files for play');
    } finally { setPlayLoading(null); }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Game Requests</h1>
        <p className="text-gray-400 mt-1">Review and manage game submissions</p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-6">
          <AlertCircle size={14} className="text-red-400" />
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-16">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      ) : requests.length === 0 ? (
        <div className="text-center py-16">
          <Gamepad2 size={40} className="mx-auto text-gray-600 mb-3" />
          <p className="text-gray-400 text-lg">No requests</p>
          <p className="text-gray-600 text-sm mt-1">Submissions will appear here.</p>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden divide-y divide-gray-800/50">
          {requests.map((req) => {
            const isOpen = expandedId === req.id;
            return (
              <div key={req.id}>
                {/* Collapsed row */}
                <button
                  onClick={() => toggle(req.id)}
                  className={`w-full flex items-center gap-3 px-5 py-3.5 text-left transition-colors ${isOpen ? 'bg-gray-800/60' : 'hover:bg-gray-800/30'}`}
                >
                  <span className="text-gray-500">{isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</span>
                  <span className="font-mono font-bold text-white text-sm min-w-[60px]">{req.game_id}</span>
                  <StatusBadge status={req.status} />
                  {req.requester_name && (
                    <span className="flex items-center gap-1 text-xs text-gray-400">
                      <User size={11} /> {req.requester_name}
                    </span>
                  )}
                  {req.game_owner_name && req.game_owner_name !== req.requester_name && (
                    <span className="text-xs text-gray-500">by {req.game_owner_name}</span>
                  )}
                  <span className="ml-auto text-[11px] text-gray-600">
                    {new Date(req.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                  </span>
                </button>

                {/* Expanded details */}
                <AnimatePresence>
                {isOpen && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                  <div className="bg-gray-800/30 border-t border-gray-700/50 px-6 py-5 space-y-4">
                    {/* Rejection note */}
                    {req.status === 'rejected' && req.admin_note && (
                      <div className="px-3 py-2 bg-red-500/5 border border-red-500/20 rounded-lg">
                        <p className="text-[10px] text-red-400 uppercase tracking-wider mb-0.5">Rejection Note</p>
                        <p className="text-xs text-gray-300">{req.admin_note}</p>
                      </div>
                    )}

                    {/* Info grid */}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Requester</p>
                        <div className="space-y-1">
                          <p className="flex items-center gap-1.5 text-gray-300"><User size={12} className="text-gray-500" /> {req.requester_name}</p>
                          {req.requester_email && <p className="flex items-center gap-1.5 text-gray-400"><Mail size={12} className="text-gray-500" /> {req.requester_email}</p>}
                        </div>
                      </div>
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Owner</p>
                        <p className="text-gray-300">{req.game_owner_name || '--'}</p>
                      </div>
                    </div>

                    {/* Description */}
                    {req.description && (
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Description</p>
                        <p className="text-sm text-gray-300 bg-gray-900/50 rounded-lg p-3">{req.description}</p>
                      </div>
                    )}

                    {/* Rules */}
                    {req.game_rules && (
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Game Rules</p>
                        <p className="text-sm text-gray-300 bg-gray-900/50 rounded-lg p-3 whitespace-pre-wrap">{req.game_rules}</p>
                      </div>
                    )}

                    {/* Message */}
                    {req.message && (
                      <div className="flex items-start gap-2 text-sm text-gray-400 bg-gray-900/50 rounded-lg p-3">
                        <MessageSquare size={14} className="text-gray-500 shrink-0 mt-0.5" />
                        <p>{req.message}</p>
                      </div>
                    )}

                    {/* Links */}
                    {(req.game_drive_link || req.game_video_link) && (
                      <div className="flex flex-wrap gap-2">
                        {req.game_drive_link && (
                          <a href={req.game_drive_link} target="_blank" rel="noopener noreferrer"
                            className="flex items-center gap-1 text-blue-400 hover:text-blue-300 bg-blue-500/10 rounded-lg px-2.5 py-1.5 text-xs transition-colors">
                            <FolderOpen size={12} /> Drive <ExternalLink size={10} />
                          </a>
                        )}
                        {req.game_video_link && (
                          <a href={req.game_video_link} target="_blank" rel="noopener noreferrer"
                            className="flex items-center gap-1 text-purple-400 hover:text-purple-300 bg-purple-500/10 rounded-lg px-2.5 py-1.5 text-xs transition-colors">
                            <Video size={12} /> Video <ExternalLink size={10} />
                          </a>
                        )}
                      </div>
                    )}

                    {/* Inline reject form */}
                    {rejectingId === req.id && (
                      <div className="flex items-center gap-2">
                        <input
                          type="text" placeholder="Rejection note (optional)" value={rejectNote}
                          onChange={(e) => setRejectNote(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Enter') openConfirm({ title: `Reject "${req.game_id}"?`, message: 'This request will be marked as rejected.', variant: 'warning', confirmText: 'Reject', onConfirm: () => { closeConfirm(); handleReject(req.id); } }); if (e.key === 'Escape') { setRejectingId(null); setRejectNote(''); } }}
                          autoFocus
                          className="flex-1 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-red-500/50"
                        />
                        <button onClick={() => openConfirm({ title: `Reject "${req.game_id}"?`, message: 'Marked as rejected.', variant: 'warning', confirmText: 'Reject', onConfirm: () => { closeConfirm(); handleReject(req.id); } })} className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm">Confirm</button>
                        <button onClick={() => { setRejectingId(null); setRejectNote(''); }} className="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-400 rounded-lg text-sm">Cancel</button>
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-gray-700/50">
                      {/* Play */}
                      <button
                        onClick={() => handlePlay(req)}
                        disabled={playLoading === req.id}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                      >
                        {playLoading === req.id ? <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-blue-400" /> : <Play size={14} />}
                        Play Test
                      </button>

                      {req.status === 'pending' && (
                        <>
                          <button
                            onClick={() => openConfirm({ title: `Approve "${req.game_id}"?`, message: 'Creates game (inactive) and removes from requests.', variant: 'success', confirmText: 'Approve', onConfirm: () => { closeConfirm(); handleApprove(req.id); } })}
                            disabled={!!actionLoading[req.id]}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                          >
                            <Check size={14} /> {actionLoading[req.id] === 'approve' ? 'Approving...' : 'Approve'}
                          </button>
                          <button
                            onClick={() => { setRejectingId(req.id); setRejectNote(''); }}
                            disabled={!!actionLoading[req.id] || rejectingId === req.id}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                          >
                            <X size={14} /> Reject
                          </button>
                        </>
                      )}

                      <button onClick={() => handleViewSource(req.id)} disabled={sourceLoading === req.id}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors disabled:opacity-50">
                        {sourceLoading === req.id ? <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-gray-400" /> : <Eye size={14} />}
                        Source
                      </button>

                      <button
                        onClick={() => openConfirm({ title: 'Delete request?', message: 'Permanently removes the request. Cannot be undone.', variant: 'danger', confirmText: 'Delete', onConfirm: () => { closeConfirm(); handleDelete(req.id); } })}
                        disabled={!!actionLoading[req.id]}
                        className="flex items-center gap-1.5 px-3 py-1.5 ml-auto bg-gray-800 hover:bg-red-600/20 text-gray-500 hover:text-red-400 rounded-lg text-sm transition-colors disabled:opacity-50"
                      >
                        <Trash2 size={14} /> Delete
                      </button>
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

      <SourceModal source={sourceData} onClose={() => setSourceData(null)} />

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
