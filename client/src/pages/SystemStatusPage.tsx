import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity, Database, Server, HardDrive, Clock, Users,
  Gamepad2, ScrollText, Trash2, RefreshCw, CheckCircle2,
  AlertTriangle, Timer, Inbox, BarChart3, Cpu, MemoryStick
} from 'lucide-react';
import { systemAPI } from '../api/client';

interface SystemStatus {
  status: string;
  timestamp: string;
  server: {
    uptime_ms: number;
    uptime_human: string;
    node_version: string;
    platform: string;
    arch: string;
    hostname: string;
  };
  memory: {
    rss_mb: number;
    heap_used_mb: number;
    heap_total_mb: number;
    external_mb: number;
    system_total_mb: number;
    system_free_mb: number;
  };
  database: {
    pool: { total: number; active: number; idle: number; waiting: number; max: number };
    size: string;
    tables: {
      games: number;
      active_games: number;
      play_sessions: number;
      temp_sessions: number;
      expired_temp_sessions: number;
      users: number;
      logs: number;
      requests: number;
      analytics: number;
    };
    oldest_log: string | null;
  };
  cleanup: {
    temp_session_ttl: string;
    auto_cleanup_interval: string;
    expired_count: number;
  };
  bridges: {
    active_bridges: number;
    idle_timeout_ms: number;
    sweep_interval_ms: number;
  };
}

function StatCard({ icon: Icon, label, value, sub, color = 'blue', className = '' }: {
  icon: any; label: string; value: string | number; sub?: string; color?: string; className?: string;
}) {
  const colorMap: Record<string, { bg: string; border: string; iconBg: string; iconText: string }> = {
    blue: { bg: '', border: 'border-gray-200 dark:border-gray-800', iconBg: 'bg-blue-500/10', iconText: 'text-blue-500' },
    green: { bg: '', border: 'border-gray-200 dark:border-gray-800', iconBg: 'bg-green-500/10', iconText: 'text-green-500' },
    amber: { bg: '', border: 'border-gray-200 dark:border-gray-800', iconBg: 'bg-amber-500/10', iconText: 'text-amber-500' },
    red: { bg: '', border: 'border-gray-200 dark:border-gray-800', iconBg: 'bg-red-500/10', iconText: 'text-red-500' },
    purple: { bg: '', border: 'border-gray-200 dark:border-gray-800', iconBg: 'bg-purple-500/10', iconText: 'text-purple-500' },
  };
  const c = colorMap[color] || colorMap.blue;
  return (
    <div className={`bg-white dark:bg-gray-900 border ${c.border} rounded-xl p-4 ${className}`}>
      <div className="flex items-center gap-3 mb-3">
        <div className={`w-9 h-9 rounded-lg ${c.iconBg} flex items-center justify-center`}>
          <Icon size={18} className={c.iconText} />
        </div>
        <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">{label}</p>
      </div>
      <p className="text-2xl font-bold text-gray-900 dark:text-white tabular-nums">{value}</p>
      {sub && <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-1">{sub}</p>}
    </div>
  );
}

function PoolBar({ used, max }: { used: number; max: number }) {
  const pct = max > 0 ? Math.round((used / max) * 100) : 0;
  const barColor = pct > 80 ? 'bg-red-500' : pct > 50 ? 'bg-amber-500' : 'bg-green-500';
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full ${barColor} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500 dark:text-gray-400 tabular-nums w-12 text-right">{pct}%</span>
    </div>
  );
}

function TableRow({ label, icon: Icon, count, className = '' }: { label: string; icon: any; count: number; className?: string }) {
  return (
    <div className={`flex items-center justify-between py-2.5 ${className}`}>
      <div className="flex items-center gap-2.5">
        <Icon size={15} className="text-gray-400 dark:text-gray-500" />
        <span className="text-sm text-gray-700 dark:text-gray-300">{label}</span>
      </div>
      <span className="text-sm font-semibold text-gray-900 dark:text-white tabular-nums">{count.toLocaleString()}</span>
    </div>
  );
}

export default function SystemStatusPage() {
  const [data, setData] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [cleaning, setCleaning] = useState(false);
  const [killingBridges, setKillingBridges] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchStatus = useCallback(async () => {
    try {
      setError('');
      const res = await systemAPI.status();
      setData(res.data);
      setLastRefresh(new Date());
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleCleanup = async () => {
    if (!confirm('Purge all expired temp sessions?')) return;
    setCleaning(true);
    try {
      const res = await systemAPI.cleanupExpired();
      alert(`Purged ${res.data.deleted} expired sessions`);
      fetchStatus();
    } catch (err: any) {
      alert('Failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setCleaning(false);
    }
  };

  const handleKillBridges = async () => {
    if (!confirm('Kill ALL active game bridges? Players will lose their sessions.')) return;
    setKillingBridges(true);
    try {
      const res = await systemAPI.killBridges();
      alert(`Killed ${res.data.killed} active bridge(s)`);
      fetchStatus();
    } catch (err: any) {
      alert('Failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setKillingBridges(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <AlertTriangle size={40} className="text-red-400 mb-3" />
        <p className="text-red-400 text-sm">{error}</p>
        <button onClick={fetchStatus} className="mt-3 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700">
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  const memUsedPct = data.memory.system_total_mb > 0
    ? Math.round(((data.memory.system_total_mb - data.memory.system_free_mb) / data.memory.system_total_mb) * 100)
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <Activity size={20} className="text-emerald-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">System Status</h1>
            <p className="text-xs text-gray-400 dark:text-gray-500">Auto-refresh every 30s · Last: {lastRefresh.toLocaleTimeString()}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/10 border border-green-500/20">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-green-500">Healthy</span>
          </div>
          <button onClick={fetchStatus}
            className="p-2 rounded-lg text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4">
        <StatCard icon={Clock} label="Uptime" value={data.server.uptime_human} sub={`Node ${data.server.node_version}`} color="green" />
        <StatCard icon={Database} label="DB Size" value={data.database.size} sub={`Pool: ${data.database.pool.total} connections (${data.database.pool.idle} idle)`} color="blue" />
        <StatCard icon={MemoryStick} label="RAM Usage" value={`${data.memory.rss_mb} MB`} sub={`Process RSS (heap: ${data.memory.heap_used_mb} MB)`} color="red" />
        <StatCard icon={MemoryStick} label="Heap Used" value={`${data.memory.heap_used_mb} MB`} sub={`of ${data.memory.heap_total_mb} MB allocated`} color="purple" />
        <StatCard icon={Cpu} label="System Memory" value={`${memUsedPct}%`} sub={`${data.memory.system_free_mb.toLocaleString()} MB free of ${data.memory.system_total_mb.toLocaleString()} MB`} color="amber" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Database size={16} className="text-blue-500" />
            Database Pool
          </h2>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1.5">
                <span>Allocated Connections</span>
                <span className="font-medium">{data.database.pool.total} / {data.database.pool.max}</span>
              </div>
              <PoolBar used={data.database.pool.total} max={data.database.pool.max} />
            </div>
            <div className="grid grid-cols-4 gap-3 pt-2 border-t border-gray-100 dark:border-gray-800">
              <div>
                <p className="text-lg font-bold text-gray-900 dark:text-white">{data.database.pool.total}</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 uppercase">Allocated</p>
              </div>
              <div>
                <p className="text-lg font-bold text-blue-500">{data.database.pool.active}</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 uppercase">Active</p>
              </div>
              <div>
                <p className="text-lg font-bold text-green-500">{data.database.pool.idle}</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 uppercase">Idle</p>
              </div>
              <div>
                <p className="text-lg font-bold text-amber-500">{data.database.pool.waiting}</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-500 uppercase">Waiting</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <HardDrive size={16} className="text-purple-500" />
            Table Records
          </h2>
          <div className="divide-y divide-gray-100 dark:divide-gray-800">
            <TableRow icon={Gamepad2} label="Games" count={data.database.tables.games} />
            <TableRow icon={CheckCircle2} label="Active Games" count={data.database.tables.active_games} />
            <TableRow icon={BarChart3} label="Play Sessions" count={data.database.tables.play_sessions} />
            <TableRow icon={Users} label="Users" count={data.database.tables.users} />
            <TableRow icon={ScrollText} label="Logs" count={data.database.tables.logs} />
            <TableRow icon={Inbox} label="Requests" count={data.database.tables.requests} />
            <TableRow icon={BarChart3} label="Analytics" count={data.database.tables.analytics} />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Timer size={16} className="text-amber-500" />
            Session Cleanup
          </h2>
          <div className="space-y-4">
            <div className="space-y-2.5">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Temp Sessions</span>
                <span className="text-sm font-semibold text-gray-900 dark:text-white">{data.database.tables.temp_sessions}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Expired</span>
                <span className={`text-sm font-semibold ${data.cleanup.expired_count > 0 ? 'text-amber-500' : 'text-green-500'}`}>
                  {data.cleanup.expired_count}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">TTL</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{data.cleanup.temp_session_ttl}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Auto-Cleanup</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Every {data.cleanup.auto_cleanup_interval}</span>
              </div>
            </div>

            <div className="pt-3 border-t border-gray-100 dark:border-gray-800">
              <button
                onClick={handleCleanup}
                disabled={cleaning || data.cleanup.expired_count === 0}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Trash2 size={15} />
                {cleaning ? 'Purging...' : `Purge ${data.cleanup.expired_count} Expired`}
              </button>
            </div>
          </div>
        </div>

        {data.bridges && (
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Gamepad2 size={16} className="text-green-500" />
            Active Game Processes
          </h2>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-xl bg-green-500/10 flex items-center justify-center">
                <span className={`text-3xl font-bold ${data.bridges.active_bridges > 0 ? 'text-green-500' : 'text-gray-400 dark:text-gray-500'}`}>
                  {data.bridges.active_bridges}
                </span>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">Python Subprocesses</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Each active game session runs a Python bridge</p>
              </div>
            </div>
            <div className="space-y-2.5 pt-3 border-t border-gray-100 dark:border-gray-800">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Idle Timeout</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{Math.round(data.bridges.idle_timeout_ms / 60000)} min</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Sweep Interval</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{Math.round(data.bridges.sweep_interval_ms / 1000)}s</span>
              </div>
              {data.bridges.active_bridges > 0 && (
                <button
                  onClick={handleKillBridges}
                  disabled={killingBridges}
                  className="w-full mt-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 hover:bg-red-500/20 disabled:opacity-50 transition-colors"
                >
                  {killingBridges ? 'Killing...' : `Kill All ${data.bridges.active_bridges} Bridge(s)`}
                </button>
              )}
            </div>
          </div>
        </div>
        )}
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Server size={16} className="text-green-500" />
          Server Info
        </h2>
        <div className="grid grid-cols-4 gap-6">
          {[
            { label: 'Platform', value: `${data.server.platform} (${data.server.arch})` },
            { label: 'Hostname', value: data.server.hostname },
            { label: 'Node.js', value: data.server.node_version },
            { label: 'Process RSS', value: `${data.memory.rss_mb} MB` },
          ].map(item => (
            <div key={item.label}>
              <p className="text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-1">{item.label}</p>
              <p className="text-sm font-medium text-gray-900 dark:text-white">{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
