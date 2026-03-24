import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { analyticsAPI } from '../api/client';
import {
  Gamepad2, Activity, Users, Zap, Trophy, ChevronRight,
  TrendingUp, Inbox, Skull, BarChart3, Target, Clock, Video, Settings
} from 'lucide-react';

function MiniBarChart({ data, color = '#3b82f6', height = 60 }) {
  const max = Math.max(...data, 1);
  return (
    <div className="flex items-end gap-[3px]" style={{ height }}>
      {data.map((val, i) => (
        <div
          key={i}
          className="flex-1 rounded-t-sm transition-all duration-300 min-w-[4px]"
          style={{
            height: `${Math.max((val / max) * 100, 4)}%`,
            backgroundColor: color,
            opacity: i === data.length - 1 ? 1 : 0.5 + (i / data.length) * 0.5,
          }}
        />
      ))}
    </div>
  );
}

function DonutChart({ value, total, color = '#3b82f6', size = 80, strokeWidth = 8 }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const percentage = total > 0 ? value / total : 0;
  const offset = circumference - percentage * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={strokeWidth} />
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
          className="transition-all duration-1000 ease-out" />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold text-white">{Math.round(percentage * 100)}%</span>
      </div>
    </div>
  );
}

function HorizontalBar({ label, value, max, color = 'bg-blue-500' }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-300 truncate max-w-[150px]">{label}</span>
        <span className="text-gray-500 font-mono ml-2">{value}</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-700 ease-out`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recordingEnabled, setRecordingEnabled] = useState(true);
  const [togglingRecording, setTogglingRecording] = useState(false);

  useEffect(() => {
    analyticsAPI.dashboard()
      .then((res) => setStats(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load dashboard'))
      .finally(() => setLoading(false));
    // Load recording setting
    fetch('/api/settings/recording_enabled')
      .then(r => r.json())
      .then(data => setRecordingEnabled(data.value === 'true'))
      .catch(() => {});
  }, []);

  const toggleRecording = async () => {
    const newVal = !recordingEnabled;
    setRecordingEnabled(newVal); // optimistic update
    setTogglingRecording(true);
    try {
      const res = await fetch('/api/settings/recording_enabled', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('arc_token')}` },
        body: JSON.stringify({ value: String(newVal) }),
      });
      const data = await res.json();
      // Confirm from server response
      setRecordingEnabled(data.value === 'true');
    } catch {
      // Revert on failure
      setRecordingEnabled(!newVal);
    } finally {
      setTogglingRecording(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400">{error}</p>
      </div>
    );
  }

  const s = stats || {};
  const maxGamePlays = Math.max(...(s.game_distribution || []).map(g => g.plays), 1);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-500 mt-1 text-sm">Platform overview and analytics</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="px-3 py-1.5 bg-gray-800 rounded-lg border border-gray-700">
            <span className="text-xs text-gray-400">Last 7 days</span>
          </div>
        </div>
      </div>

      {/* Top stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'Total Games', value: s.total_games, icon: Gamepad2, color: 'blue', sub: `${s.active_games} active` },
          { label: 'Total Plays', value: s.total_plays, icon: Activity, color: 'purple', sub: `${s.total_wins} wins` },
          { label: 'Win Rate', value: `${s.win_rate}%`, icon: TrendingUp, color: 'green', sub: `${s.total_game_overs} game overs` },
          { label: 'Users', value: s.total_users, icon: Users, color: 'orange', sub: `${s.pending_requests} pending requests` },
        ].map(({ label, value, icon: Icon, color, sub }, index) => (
          <motion.div key={label} className="bg-gray-900 rounded-xl border border-gray-800 p-5 group hover:border-gray-700 transition-colors"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="flex items-start justify-between mb-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                color === 'blue' ? 'bg-blue-500/10 text-blue-400' :
                color === 'purple' ? 'bg-purple-500/10 text-purple-400' :
                color === 'green' ? 'bg-emerald-500/10 text-emerald-400' :
                'bg-orange-500/10 text-orange-400'
              }`}>
                <Icon size={20} />
              </div>
            </div>
            <p className="text-3xl font-bold text-white tracking-tight">{value}</p>
            <p className="text-xs text-gray-500 mt-1">{label}</p>
            <p className={`text-[11px] mt-0.5 ${
              color === 'blue' ? 'text-blue-400/60' :
              color === 'purple' ? 'text-purple-400/60' :
              color === 'green' ? 'text-emerald-400/60' :
              'text-orange-400/60'
            }`}>{sub}</p>
          </motion.div>
        ))}
      </div>

      {/* App Settings */}
      <div className="mb-6 bg-gray-900 rounded-xl border border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gray-800 border border-gray-700">
              <Settings size={16} className="text-gray-400" />
            </div>
            <div>
              <h3 className="text-sm font-medium text-white">App Settings</h3>
              <p className="text-[11px] text-gray-500">Platform-wide toggles</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* Recording toggle */}
            <div className="flex items-center gap-3">
              <Video size={14} className={recordingEnabled ? 'text-red-400' : 'text-gray-600'} />
              <span className="text-xs text-gray-400">Video Recording</span>
              <button
                onClick={toggleRecording}
                disabled={togglingRecording}
                className={`relative w-11 h-6 rounded-full transition-colors ${recordingEnabled ? 'bg-red-600' : 'bg-gray-700'}`}
              >
                <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${recordingEnabled ? 'translate-x-5' : 'translate-x-0'}`} />
              </button>
              <span className={`text-[10px] font-mono ${recordingEnabled ? 'text-red-400' : 'text-gray-600'}`}>
                {recordingEnabled ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {/* Daily plays chart */}
        <motion.div className="lg:col-span-2 bg-gray-900 rounded-xl border border-gray-800 p-5"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="text-sm font-semibold text-white">Play Activity</h3>
              <p className="text-xs text-gray-500 mt-0.5">Sessions over the last 7 days</p>
            </div>
            <div className="flex items-center gap-4 text-[11px]">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                <span className="text-gray-400">Plays</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span className="text-gray-400">Wins</span>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="relative h-[160px]">
            <div className="absolute inset-0 flex items-end gap-[6px]">
              {(s.daily_plays || []).map((plays, i) => {
                const wins = (s.daily_wins || [])[i] || 0;
                const max = Math.max(...(s.daily_plays || [1]), 1);
                return (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1 h-full justify-end">
                    <div className="w-full flex gap-[2px] items-end" style={{ height: '85%' }}>
                      <div className="flex-1 bg-blue-500/80 rounded-t-sm transition-all duration-500"
                        style={{ height: `${Math.max((plays / max) * 100, 2)}%` }} />
                      <div className="flex-1 bg-emerald-500/80 rounded-t-sm transition-all duration-500"
                        style={{ height: `${Math.max((wins / max) * 100, 2)}%` }} />
                    </div>
                    <span className="text-[9px] text-gray-600">{(s.daily_labels || [])[i]}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* Win rate donut */}
        <motion.div className="bg-gray-900 rounded-xl border border-gray-800 p-5 flex flex-col items-center justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-sm font-semibold text-white mb-1">Win Rate</h3>
          <p className="text-xs text-gray-500 mb-5">Overall completion rate</p>
          <DonutChart value={s.total_wins || 0} total={s.total_plays || 0} color="#10b981" size={110} strokeWidth={10} />
          <div className="flex items-center gap-6 mt-5 text-xs">
            <div className="text-center">
              <p className="text-emerald-400 font-bold text-lg">{s.total_wins || 0}</p>
              <p className="text-gray-500">Wins</p>
            </div>
            <div className="w-px h-8 bg-gray-800" />
            <div className="text-center">
              <p className="text-red-400 font-bold text-lg">{s.total_game_overs || 0}</p>
              <p className="text-gray-500">Game Overs</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Bottom row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Game distribution */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Game Popularity</h3>
          {(s.game_distribution || []).length > 0 ? (
            <div className="space-y-3">
              {s.game_distribution.map((g) => (
                <HorizontalBar key={g.name} label={g.name} value={g.plays} max={maxGamePlays} color="bg-blue-500" />
              ))}
            </div>
          ) : (
            <p className="text-gray-600 text-sm">No play data yet</p>
          )}
        </div>

        {/* Recent games */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Recent Games</h3>
            <Link to="/admin/games" className="text-[11px] text-blue-400 hover:text-blue-300 flex items-center gap-0.5">
              View all <ChevronRight size={12} />
            </Link>
          </div>
          <div className="space-y-2">
            {(s.recent_games || []).length > 0 ? s.recent_games.map((game) => (
              <Link
                key={game.id || game.game_id}
                to={`/admin/games/${game.game_id}`}
                className="flex items-center justify-between p-2.5 rounded-lg bg-gray-800/40 hover:bg-gray-800 transition-colors"
              >
                <div className="min-w-0">
                  <p className="text-xs font-medium text-white truncate">{game.name || game.game_id}</p>
                  <p className="text-[10px] text-gray-500 font-mono">{game.game_id}</p>
                </div>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ml-2 ${
                  game.is_active ? 'bg-emerald-500/15 text-emerald-400' : 'bg-gray-700 text-gray-400'
                }`}>
                  {game.is_active ? 'Active' : 'Inactive'}
                </span>
              </Link>
            )) : (
              <p className="text-gray-600 text-sm">No games yet</p>
            )}
          </div>
        </div>

        {/* Top played */}
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Top Played</h3>
            <Trophy size={16} className="text-yellow-400" />
          </div>
          <div className="space-y-2">
            {(s.top_played_games || []).length > 0 ? s.top_played_games.map((game, idx) => (
              <div key={game.game_id} className="flex items-center gap-3 p-2.5 rounded-lg bg-gray-800/40">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                  idx === 0 ? 'bg-yellow-500/20 text-yellow-400' :
                  idx === 1 ? 'bg-gray-400/20 text-gray-300' :
                  idx === 2 ? 'bg-orange-500/20 text-orange-400' :
                  'bg-gray-800 text-gray-500'
                }`}>
                  {idx + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-white truncate">{game.name}</p>
                  <div className="flex items-center gap-2 text-[10px] text-gray-500">
                    <span>{game.total_plays} plays</span>
                    <span className="text-emerald-400">{game.total_wins} wins</span>
                  </div>
                </div>
              </div>
            )) : (
              <p className="text-gray-600 text-sm">No play data yet</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
