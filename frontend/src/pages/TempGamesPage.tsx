import React, { useState, useEffect } from 'react';
import { analyticsAPI } from '../api/client';
import ConfirmModal from '../components/ConfirmModal';
import {
  Clock, User, Gamepad2, ChevronDown, ChevronRight,
  Layers, Skull, Activity, Timer, Trash2
} from 'lucide-react';

function formatTime(seconds) {
  if (!seconds) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function timeAgo(dateStr) {
  if (!dateStr) return '';
  const diff = (Date.now() - new Date(dateStr).getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function TempGamesPage() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);
  const [showDeleteAll, setShowDeleteAll] = useState(false);

  const fetchSessions = () => {
    setLoading(true);
    analyticsAPI.tempSessions()
      .then((res) => setSessions(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchSessions(); }, []);

  const handleDeleteAll = async () => {
    try {
      await analyticsAPI.deleteTempSessions();
      setSessions([]);
    } catch {
      // silently ignored
    }
    setShowDeleteAll(false);
  };

  const toggle = (id) => setExpandedId(prev => prev === id ? null : id);

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Temp Games</h1>
          <p className="text-gray-400 mt-1">
            "Play Your Own" sessions -- auto-deleted after 24 hours
          </p>
        </div>
        {sessions.length > 0 && (
          <button
            onClick={() => setShowDeleteAll(true)}
            className="flex items-center gap-2 px-4 py-2 bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 text-red-400 rounded-lg text-sm transition-colors"
          >
            <Trash2 size={14} /> Clear All Logs
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      ) : sessions.length === 0 ? (
        <div className="text-center py-16">
          <Gamepad2 size={40} className="mx-auto text-gray-600 mb-3" />
          <p className="text-gray-400 text-lg">No temp game sessions</p>
          <p className="text-gray-600 text-sm mt-1">When users play their own games, sessions will appear here.</p>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
          <div className="px-6 py-3 border-b border-gray-800 bg-gray-800/50 flex items-center justify-between">
            <span className="text-sm font-medium text-gray-400">{sessions.length} session{sessions.length !== 1 ? 's' : ''}</span>
            <span className="text-xs text-gray-500">Data retained for 24 hours</span>
          </div>

          <div className="divide-y divide-gray-800/50">
            {sessions.map((s) => {
              const isOpen = expandedId === s.id;
              const levelStats = s.level_stats || [];
              const actionLog = s.action_log || [];

              return (
                <div key={s.id}>
                  {/* Row */}
                  <button
                    onClick={() => toggle(s.id)}
                    className={`w-full flex items-center gap-3 px-6 py-3.5 text-left transition-colors ${
                      isOpen ? 'bg-gray-800/60' : 'hover:bg-gray-800/30'
                    }`}
                  >
                    <span className="text-gray-500">
                      {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </span>

                    {/* Player */}
                    <div className="flex items-center gap-2 min-w-[120px]">
                      <User size={14} className="text-gray-500" />
                      <span className="text-white text-sm font-medium truncate">{s.player_name}</span>
                    </div>

                    {/* Game ID */}
                    <span className="text-blue-400 font-mono text-xs bg-blue-500/10 px-2 py-0.5 rounded">
                      {s.game_id}
                    </span>

                    {/* State */}
                    <span className={`px-2 py-0.5 rounded text-[11px] font-medium shrink-0 ${
                      s.state === 'WIN' ? 'bg-green-500/20 text-green-400' :
                      s.state === 'GAME_OVER' ? 'bg-red-500/20 text-red-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {s.state === 'WIN' ? 'WIN' : s.state === 'GAME_OVER' ? 'GAME OVER' : 'PLAYING'}
                    </span>

                    {/* Quick stats */}
                    <div className="flex items-center gap-4 ml-auto text-xs text-gray-400">
                      <span className="flex items-center gap-1 font-mono">
                        <Layers size={12} className="text-blue-400" /> Lv {s.current_level + 1}
                      </span>
                      <span className="flex items-center gap-1 font-mono">
                        <Clock size={12} className="text-blue-400" /> {formatTime(s.total_time)}
                      </span>
                      <span className="flex items-center gap-1 font-mono">
                        <Activity size={12} className="text-green-400" /> {s.total_actions}
                      </span>
                      <span className="flex items-center gap-1 font-mono">
                        <Skull size={12} className="text-red-400" /> {s.game_overs}
                      </span>
                      <span className="text-gray-600 hidden sm:inline">
                        {timeAgo(s.started_at)}
                      </span>
                    </div>
                  </button>

                  {/* Expanded */}
                  {isOpen && (
                    <div className="bg-gray-800/40 border-t border-gray-700/50 px-6 py-4 space-y-4">
                      {/* Summary */}
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                        {[
                          { label: 'Total Time', value: formatTime(s.total_time) },
                          { label: 'Actions', value: s.total_actions },
                          { label: 'Level', value: `Lv ${s.current_level + 1}` },
                          { label: 'Game Overs', value: s.game_overs, color: 'text-red-400' },
                          { label: 'Score', value: s.score },
                        ].map((stat, i) => (
                          <div key={i} className="bg-gray-900/60 rounded-lg p-3">
                            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{stat.label}</p>
                            <p className={`font-mono text-sm font-bold ${stat.color || 'text-white'}`}>{stat.value}</p>
                          </div>
                        ))}
                      </div>

                      {/* Level Stats */}
                      {levelStats.length > 0 && (
                        <div>
                          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Per-Level Breakdown</p>
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-gray-700/50">
                                <th className="text-left py-2 px-3 text-gray-500">Level</th>
                                <th className="text-right py-2 px-3 text-gray-500">Time</th>
                                <th className="text-right py-2 px-3 text-gray-500">Moves</th>
                                <th className="text-right py-2 px-3 text-gray-500">Deaths</th>
                                <th className="text-right py-2 px-3 text-gray-500">Lives</th>
                                <th className="text-right py-2 px-3 text-gray-500">Resets</th>
                                <th className="text-right py-2 px-3 text-gray-500">Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {levelStats.map((ls, i) => (
                                <tr key={i} className="border-b border-gray-800/50 last:border-0">
                                  <td className="py-2 px-3 font-mono font-medium text-gray-300">Lv {ls.level + 1}</td>
                                  <td className="py-2 px-3 text-right font-mono text-gray-300">{formatTime(ls.time)}</td>
                                  <td className="py-2 px-3 text-right font-mono text-gray-300">{ls.actions}</td>
                                  <td className="py-2 px-3 text-right font-mono text-red-400">{ls.game_overs || 0}</td>
                                  <td className="py-2 px-3 text-right font-mono text-gray-300">{ls.lives_used || 0}</td>
                                  <td className="py-2 px-3 text-right font-mono text-gray-300">{ls.resets || 0}</td>
                                  <td className="py-2 px-3 text-right">
                                    {ls.completed
                                      ? <span className="px-1.5 py-0.5 rounded bg-green-500/15 text-green-400 text-[10px]">Done</span>
                                      : <span className="px-1.5 py-0.5 rounded bg-gray-700 text-gray-400 text-[10px]">Incomplete</span>
                                    }
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Action Log */}
                      {actionLog.length > 0 && (
                        <div>
                          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">
                            Action Log ({actionLog.length} actions)
                          </p>
                          <div className="max-h-48 overflow-y-auto bg-gray-900/60 rounded-lg p-3">
                            <div className="flex flex-wrap gap-1">
                              {actionLog.map((a, i) => (
                                <span
                                  key={i}
                                  className="px-1.5 py-0.5 rounded bg-gray-800 text-[10px] font-mono text-gray-400"
                                  title={`Level ${a.level + 1} @ ${formatTime(a.time)}`}
                                >
                                  {a.action}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Timestamps */}
                      <div className="flex items-center gap-4 pt-2 border-t border-gray-700/50 text-[11px] text-gray-500">
                        <span>Started: {s.started_at ? new Date(s.started_at).toLocaleString() : '--'}</span>
                        {s.ended_at && <span>Ended: {new Date(s.ended_at).toLocaleString()}</span>}
                        <span className="ml-auto">Expires: {s.expires_at ? new Date(s.expires_at).toLocaleString() : '--'}</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <ConfirmModal
        open={showDeleteAll}
        onClose={() => setShowDeleteAll(false)}
        onConfirm={handleDeleteAll}
        title="Clear all temp game logs?"
        message={`This will permanently delete all ${sessions.length} session log(s). This cannot be undone.`}
        confirmText="Delete All"
        variant="danger"
      />
    </div>
  );
}
