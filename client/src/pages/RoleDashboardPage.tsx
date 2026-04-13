import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { approvalAPI, teamsAPI, analyticsAPI } from '../api/client';
import { motion } from 'framer-motion';
import {
  Gamepad2, Upload, ClipboardCheck, Users, Clock, CheckCircle2,
  XCircle, AlertCircle, ArrowRight, TrendingUp, Activity, Trophy, ChevronRight
} from 'lucide-react';

const APPROVAL_STATUS_LABELS = {
  draft: 'Draft',
  pending_ql: 'Pending QL Review',
  ql_approved: 'QL Approved',
  pending_pl: 'Pending PL Review',
  approved: 'Approved',
  rejected: 'Rejected',
};

const STATUS_COLORS = {
  draft: 'text-gray-400 bg-gray-500/10',
  pending_ql: 'text-yellow-400 bg-yellow-500/10',
  ql_approved: 'text-blue-400 bg-blue-500/10',
  pending_pl: 'text-purple-400 bg-purple-500/10',
  approved: 'text-green-400 bg-green-500/10',
  rejected: 'text-red-400 bg-red-500/10',
};

function StatCard({ icon: Icon, label, value, color, onClick = undefined }) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      onClick={onClick}
      className={`bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5 ${onClick ? 'cursor-pointer hover:border-blue-500/30' : ''}`}
    >
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${color}`}>
          <Icon size={20} />
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
        </div>
      </div>
    </motion.div>
  );
}

function GameRow({ game, showActions, onAction }) {
  return (
    <div className="flex items-center gap-4 px-4 py-3 border-b border-gray-100 dark:border-gray-800/50 hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors">
      <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-gray-800 flex items-center justify-center shrink-0">
        <Gamepad2 size={18} className="text-gray-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">{game.game_id}</p>
        <p className="text-xs text-gray-500 truncate">{game.name || game.game_id}</p>
      </div>
      <span className={`px-2.5 py-1 rounded-full text-[11px] font-medium ${STATUS_COLORS[game.approval_status] || STATUS_COLORS.draft}`}>
        {APPROVAL_STATUS_LABELS[game.approval_status] || game.approval_status}
      </span>
      {showActions && (
        <button
          onClick={() => onAction(game)}
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          View <ArrowRight size={12} />
        </button>
      )}
    </div>
  );
}

function TaskerDashboard() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    approvalAPI.myGames()
      .then(res => setGames(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const drafts = games.filter(g => g.approval_status === 'draft');
  const pending = games.filter(g => ['pending_ql', 'ql_approved', 'pending_pl'].includes(g.approval_status));
  const approved = games.filter(g => g.approval_status === 'approved');
  const rejected = games.filter(g => g.approval_status === 'rejected');

  return (
    <>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={Gamepad2} label="Total Games" value={games.length} color="text-blue-400 bg-blue-500/10" />
        <StatCard icon={Clock} label="Pending Review" value={pending.length} color="text-yellow-400 bg-yellow-500/10" />
        <StatCard icon={CheckCircle2} label="Approved" value={approved.length} color="text-green-400 bg-green-500/10" />
        <StatCard icon={XCircle} label="Rejected" value={rejected.length} color="text-red-400 bg-red-500/10" onClick={() => navigate('/dashboard/my-games')} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Draft Games</h3>
            <button onClick={() => navigate('/dashboard/upload')} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              <Upload size={12} /> Upload New
            </button>
          </div>
          {loading ? (
            <div className="py-8 text-center text-gray-500 text-sm">Loading...</div>
          ) : drafts.length === 0 ? (
            <div className="py-8 text-center text-gray-500 text-sm">No draft games</div>
          ) : (
            drafts.slice(0, 5).map(g => <GameRow key={g.id} game={g} showActions onAction={() => navigate('/dashboard/my-games')} />)
          )}
        </div>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Needs Attention</h3>
          </div>
          {loading ? (
            <div className="py-8 text-center text-gray-500 text-sm">Loading...</div>
          ) : rejected.length === 0 ? (
            <div className="py-8 text-center text-gray-500 text-sm flex flex-col items-center gap-2">
              <CheckCircle2 size={24} className="text-green-400" />
              <span>All good! No rejected games.</span>
            </div>
          ) : (
            rejected.slice(0, 5).map(g => <GameRow key={g.id} game={g} showActions onAction={() => navigate('/dashboard/my-games')} />)
          )}
        </div>
      </div>
    </>
  );
}

function QLDashboard() {
  const [queue, setQueue] = useState([]);
  const [team, setTeam] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([
      approvalAPI.qlQueue().catch(() => ({ data: [] })),
      teamsAPI.myTeam().catch(() => ({ data: [] })),
    ]).then(([qRes, tRes]) => {
      setQueue(qRes.data);
      setTeam(tRes.data);
    }).finally(() => setLoading(false));
  }, []);

  const pendingReview = queue.filter(g => g.approval_status === 'pending_ql');
  const approved = queue.filter(g => ['ql_approved', 'pending_pl', 'approved'].includes(g.approval_status));

  return (
    <>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={ClipboardCheck} label="Pending Review" value={pendingReview.length} color="text-yellow-400 bg-yellow-500/10" onClick={() => navigate('/dashboard/review')} />
        <StatCard icon={CheckCircle2} label="Approved" value={approved.length} color="text-green-400 bg-green-500/10" />
        <StatCard icon={Users} label="Team Size" value={team.length} color="text-blue-400 bg-blue-500/10" onClick={() => navigate('/dashboard/team')} />
        <StatCard icon={Gamepad2} label="Total in Queue" value={queue.length} color="text-purple-400 bg-purple-500/10" />
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Pending Review</h3>
          <button onClick={() => navigate('/dashboard/review')} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
            View All <ArrowRight size={12} />
          </button>
        </div>
        {loading ? (
          <div className="py-8 text-center text-gray-500 text-sm">Loading...</div>
        ) : pendingReview.length === 0 ? (
          <div className="py-8 text-center text-gray-500 text-sm flex flex-col items-center gap-2">
            <CheckCircle2 size={24} className="text-green-400" />
            <span>No games pending your review</span>
          </div>
        ) : (
          pendingReview.slice(0, 8).map(g => <GameRow key={g.id} game={g} showActions onAction={() => navigate('/dashboard/review')} />)
        )}
      </div>
    </>
  );
}

function PLDashboard() {
  const [queue, setQueue] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    approvalAPI.plQueue()
      .then(res => setQueue(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const pendingReview = queue.filter(g => g.approval_status === 'pending_pl');
  const approved = queue.filter(g => g.approval_status === 'approved');

  return (
    <>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard icon={ClipboardCheck} label="Pending Review" value={pendingReview.length} color="text-yellow-400 bg-yellow-500/10" onClick={() => navigate('/dashboard/review')} />
        <StatCard icon={CheckCircle2} label="Approved" value={approved.length} color="text-green-400 bg-green-500/10" />
        <StatCard icon={Gamepad2} label="Total in Queue" value={queue.length} color="text-purple-400 bg-purple-500/10" />
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Pending Final Approval</h3>
          <button onClick={() => navigate('/dashboard/review')} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
            View All <ArrowRight size={12} />
          </button>
        </div>
        {loading ? (
          <div className="py-8 text-center text-gray-500 text-sm">Loading...</div>
        ) : pendingReview.length === 0 ? (
          <div className="py-8 text-center text-gray-500 text-sm flex flex-col items-center gap-2">
            <CheckCircle2 size={24} className="text-green-400" />
            <span>No games pending final approval</span>
          </div>
        ) : (
          pendingReview.slice(0, 8).map(g => <GameRow key={g.id} game={g} showActions onAction={() => navigate('/dashboard/review')} />)
        )}
      </div>
    </>
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
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" className="stroke-gray-200 dark:stroke-white/5" strokeWidth={strokeWidth} />
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
          className="transition-all duration-1000 ease-out" />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-lg font-bold text-gray-900 dark:text-white">{Math.round(percentage * 100)}%</span>
      </div>
    </div>
  );
}

function HorizontalBar({ label, value, max, color = 'bg-blue-500' }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-700 dark:text-gray-300 truncate max-w-[150px]">{label}</span>
        <span className="text-gray-400 dark:text-gray-500 font-mono ml-2">{value}</span>
      </div>
      <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-700 ease-out`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function SuperAdminDashboard() {
  const [allGames, setAllGames] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([
      approvalAPI.allGames().catch(() => ({ data: [] })),
      analyticsAPI.dashboard().catch(() => ({ data: null })),
    ]).then(([gRes, sRes]) => {
      setAllGames(gRes.data);
      setStats(sRes.data);
    }).finally(() => setLoading(false));
  }, []);

  const pending = allGames.filter(g => ['pending_ql', 'pending_pl'].includes(g.approval_status));
  const approved = allGames.filter(g => g.approval_status === 'approved');
  const rejected = allGames.filter(g => g.approval_status === 'rejected');
  const drafts = allGames.filter(g => g.approval_status === 'draft');
  const s = stats || {};
  const maxGamePlays = Math.max(...(s.game_distribution || []).map(g => g.plays), 1);

  return (
    <>
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        <StatCard icon={Gamepad2} label="Total Games" value={allGames.length} color="text-blue-400 bg-blue-500/10" />
        <StatCard icon={AlertCircle} label="Drafts" value={drafts.length} color="text-gray-400 bg-gray-500/10" />
        <StatCard icon={Clock} label="Pending" value={pending.length} color="text-yellow-400 bg-yellow-500/10" onClick={() => navigate('/dashboard/review')} />
        <StatCard icon={CheckCircle2} label="Approved" value={approved.length} color="text-green-400 bg-green-500/10" />
        <StatCard icon={XCircle} label="Rejected" value={rejected.length} color="text-red-400 bg-red-500/10" />
      </div>

      {stats && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Total Plays', value: s.total_plays, icon: Activity, clr: 'purple', sub: `${s.total_wins} wins` },
              { label: 'Win Rate', value: `${s.win_rate}%`, icon: TrendingUp, clr: 'green', sub: `${s.total_game_overs} game overs` },
              { label: 'Active Games', value: s.active_games, icon: Gamepad2, clr: 'blue', sub: `of ${s.total_games} total` },
              { label: 'Users', value: s.total_users, icon: Users, clr: 'orange', sub: `${s.pending_requests} pending requests` },
            ].map(({ label, value, icon: Icon, clr, sub }, idx) => (
              <motion.div key={label} className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5"
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.08 }}>
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${
                  clr === 'blue' ? 'bg-blue-500/10 text-blue-400' :
                  clr === 'purple' ? 'bg-purple-500/10 text-purple-400' :
                  clr === 'green' ? 'bg-emerald-500/10 text-emerald-400' :
                  'bg-orange-500/10 text-orange-400'
                }`}>
                  <Icon size={20} />
                </div>
                <p className="text-3xl font-bold text-gray-900 dark:text-white tracking-tight">{value}</p>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{label}</p>
                <p className={`text-[11px] mt-0.5 ${
                  clr === 'blue' ? 'text-blue-400/60' :
                  clr === 'purple' ? 'text-purple-400/60' :
                  clr === 'green' ? 'text-emerald-400/60' :
                  'text-orange-400/60'
                }`}>{sub}</p>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <div className="lg:col-span-2 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Play Activity</h3>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Sessions over the last 7 days</p>
                </div>
                <div className="flex items-center gap-4 text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                    <span className="text-gray-500 dark:text-gray-400">Plays</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                    <span className="text-gray-500 dark:text-gray-400">Wins</span>
                  </div>
                </div>
              </div>
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
                        <span className="text-[9px] text-gray-400 dark:text-gray-600">{(s.daily_labels || [])[i]}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5 flex flex-col items-center justify-center">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Win Rate</h3>
              <p className="text-xs text-gray-400 dark:text-gray-500 mb-5">Overall completion rate</p>
              <DonutChart value={s.total_wins || 0} total={s.total_plays || 0} color="#10b981" size={110} strokeWidth={10} />
              <div className="flex items-center gap-6 mt-5 text-xs">
                <div className="text-center">
                  <p className="text-emerald-400 font-bold text-lg">{s.total_wins || 0}</p>
                  <p className="text-gray-400 dark:text-gray-500">Wins</p>
                </div>
                <div className="w-px h-8 bg-gray-200 dark:bg-gray-800" />
                <div className="text-center">
                  <p className="text-red-400 font-bold text-lg">{s.total_game_overs || 0}</p>
                  <p className="text-gray-400 dark:text-gray-500">Game Overs</p>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-4">Game Popularity</h3>
              {(s.game_distribution || []).length > 0 ? (
                <div className="space-y-3">
                  {s.game_distribution.map((g) => (
                    <HorizontalBar key={g.name} label={g.name} value={g.plays} max={maxGamePlays} color="bg-blue-500" />
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 dark:text-gray-600 text-sm">No play data yet</p>
              )}
            </div>

            <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Recent Games</h3>
                <Link to="/dashboard/games" className="text-[11px] text-blue-400 hover:text-blue-300 flex items-center gap-0.5">
                  View all <ChevronRight size={12} />
                </Link>
              </div>
              <div className="space-y-2">
                {(s.recent_games || []).length > 0 ? s.recent_games.map((game) => (
                  <Link
                    key={game.id || game.game_id}
                    to={`/dashboard/games/${game.game_id}`}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/40 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-gray-900 dark:text-white truncate">{game.name || game.game_id}</p>
                      <p className="text-[10px] text-gray-400 dark:text-gray-500 font-mono">{game.game_id}</p>
                    </div>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ml-2 ${
                      game.is_active ? 'bg-emerald-500/15 text-emerald-400' : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                    }`}>
                      {game.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </Link>
                )) : (
                  <p className="text-gray-400 dark:text-gray-600 text-sm">No games yet</p>
                )}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Top Played</h3>
                <Trophy size={16} className="text-yellow-400" />
              </div>
              <div className="space-y-2">
                {(s.top_played_games || []).length > 0 ? s.top_played_games.map((game, idx) => (
                  <div key={game.game_id} className="flex items-center gap-3 p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/40">
                    <span className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                      idx === 0 ? 'bg-yellow-500/20 text-yellow-400' :
                      idx === 1 ? 'bg-gray-400/20 text-gray-400 dark:text-gray-300' :
                      idx === 2 ? 'bg-orange-500/20 text-orange-400' :
                      'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500'
                    }`}>
                      {idx + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-900 dark:text-white truncate">{game.name}</p>
                      <div className="flex items-center gap-2 text-[10px] text-gray-400 dark:text-gray-500">
                        <span>{game.total_plays} plays</span>
                        <span className="text-emerald-400">{game.total_wins} wins</span>
                      </div>
                    </div>
                  </div>
                )) : (
                  <p className="text-gray-400 dark:text-gray-600 text-sm">No play data yet</p>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Pending Approval</h3>
            <button onClick={() => navigate('/dashboard/review')} className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              View All <ArrowRight size={12} />
            </button>
          </div>
          {loading ? (
            <div className="py-8 text-center text-gray-500 text-sm">Loading...</div>
          ) : pending.length === 0 ? (
            <div className="py-8 text-center text-gray-500 text-sm">No pending games</div>
          ) : (
            pending.slice(0, 5).map(g => <GameRow key={g.id} game={g} showActions onAction={() => navigate('/dashboard/review')} />)
          )}
        </div>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Quick Actions</h3>
          </div>
          <div className="p-4 space-y-2">
            <button onClick={() => navigate('/dashboard/team')} className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-left">
              <Users size={18} className="text-blue-400" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">Manage Teams</p>
                <p className="text-xs text-gray-500">Assign users to team leads</p>
              </div>
            </button>
            <button onClick={() => navigate('/dashboard/games')} className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-left">
              <TrendingUp size={18} className="text-purple-400" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">All Games</p>
                <p className="text-xs text-gray-500">View and manage all games</p>
              </div>
            </button>
            <button onClick={() => navigate('/dashboard/upload')} className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-left">
              <Upload size={18} className="text-green-400" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">Upload Game</p>
                <p className="text-xs text-gray-500">Upload and auto-approve</p>
              </div>
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

const ROLE_GREETINGS = {
  tasker: 'Ready to upload some games?',
  ql: 'Games are waiting for your review.',
  pl: 'Final approvals need your attention.',
  super_admin: 'Full system overview.',
};

export default function RoleDashboardPage() {
  const { user, isSuperAdmin, isQL, isPL } = useAuth();
  const role = user?.role || (user?.is_admin ? 'super_admin' : 'tasker');

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Welcome, {user?.username}
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          {ROLE_GREETINGS[role]}
        </p>
      </div>

      {isSuperAdmin && <SuperAdminDashboard />}
      {isQL && <QLDashboard />}
      {isPL && <PLDashboard />}
      {!isSuperAdmin && !isQL && !isPL && <TaskerDashboard />}
    </div>
  );
}
