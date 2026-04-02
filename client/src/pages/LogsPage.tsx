import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ScrollText, Trash2, Pause, Play, ArrowDown, Search, X, Radio } from 'lucide-react';
import { logsAPI } from '../api/client';

interface LogEntry {
  id: string;
  level: string;
  source: string;
  message: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

const LEVEL_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  debug: { bg: 'bg-gray-500/10', text: 'text-gray-400', dot: 'bg-gray-500' },
  info: { bg: 'bg-blue-500/10', text: 'text-blue-400', dot: 'bg-blue-500' },
  warn: { bg: 'bg-amber-500/10', text: 'text-amber-400', dot: 'bg-amber-500' },
  error: { bg: 'bg-red-500/10', text: 'text-red-400', dot: 'bg-red-500' },
};

const MAX_VISIBLE = 500;

function formatTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
    '.' + String(d.getMilliseconds()).padStart(3, '0');
}

export default function LogsPage() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [stats, setStats] = useState<{ total: number; by_level: Record<string, number>; connected_clients: number } | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [filterLevel, setFilterLevel] = useState('');
  const [filterSource, setFilterSource] = useState('');
  const [searchText, setSearchText] = useState('');
  const [streaming, setStreaming] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    logsAPI.list({ limit: 200 }).then(r => setLogs(r.data.logs.reverse()));
    logsAPI.sources().then(r => setSources(r.data));
    logsAPI.stats().then(r => setStats(r.data));
  }, []);

  useEffect(() => {
    if (!streaming) {
      esRef.current?.close();
      esRef.current = null;
      return;
    }
    const es = new EventSource(logsAPI.streamUrl());
    esRef.current = es;
    es.addEventListener('log', (e) => {
      const entry: LogEntry = JSON.parse(e.data);
      setLogs(prev => {
        const next = [...prev, entry];
        return next.length > MAX_VISIBLE ? next.slice(-MAX_VISIBLE) : next;
      });
      setStats(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          total: prev.total + 1,
          by_level: {
            ...prev.by_level,
            [entry.level]: (prev.by_level[entry.level] || 0) + 1,
          },
        };
      });
    });
    return () => { es.close(); };
  }, [streaming]);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handlePurge = useCallback(async () => {
    if (!confirm('Purge all logs?')) return;
    const r = await logsAPI.cleanup();
    alert(`Deleted ${r.data.deleted} old logs`);
    logsAPI.stats().then(r => setStats(r.data));
  }, []);

  const toggleExpand = (id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const filtered = logs.filter(l => {
    if (filterLevel && l.level !== filterLevel) return false;
    if (filterSource && l.source !== filterSource) return false;
    if (searchText && !l.message.toLowerCase().includes(searchText.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
            <ScrollText size={20} className="text-purple-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">Application Logs</h1>
            <p className="text-xs text-gray-400 dark:text-gray-500">Live stream · 2-hour retention</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setAutoScroll(!autoScroll)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${autoScroll ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-300 dark:border-gray-700'}`}>
            <ArrowDown size={14} /> Auto-scroll
          </button>
          <button onClick={() => setStreaming(!streaming)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${streaming ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-300 dark:border-gray-700'}`}>
            {streaming ? <><Radio size={14} /> Live</> : <><Pause size={14} /> Paused</>}
          </button>
          <button onClick={handlePurge}
            className="px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-colors">
            <Trash2 size={14} /> Purge
          </button>
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Total', value: stats.total, color: 'text-gray-900 dark:text-white' },
            { label: 'Errors', value: stats.by_level?.error || 0, color: 'text-red-400' },
            { label: 'Warnings', value: stats.by_level?.warn || 0, color: 'text-amber-400' },
            { label: 'SSE Clients', value: stats.connected_clients, color: 'text-green-400' },
          ].map(s => (
            <div key={s.label} className="bg-white/50 dark:bg-gray-900/50 border border-gray-200/60 dark:border-gray-800/60 rounded-xl px-4 py-3">
              <p className="text-[10px] text-gray-400 dark:text-gray-500 uppercase tracking-wider">{s.label}</p>
              <p className={`text-lg font-bold ${s.color}`}>{s.value.toLocaleString()}</p>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2">
        <select value={filterLevel} onChange={e => setFilterLevel(e.target.value)}
          className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 outline-none focus:border-gray-400 dark:focus:border-gray-600">
          <option value="">All Levels</option>
          {['debug', 'info', 'warn', 'error'].map(l => <option key={l} value={l}>{l.toUpperCase()}</option>)}
        </select>
        <select value={filterSource} onChange={e => setFilterSource(e.target.value)}
          className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 outline-none focus:border-gray-400 dark:focus:border-gray-600">
          <option value="">All Sources</option>
          {sources.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <div className="relative flex-1">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500" />
          <input value={searchText} onChange={e => setSearchText(e.target.value)} placeholder="Search logs..."
            className="w-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg pl-8 pr-8 py-1.5 text-xs text-gray-700 dark:text-gray-300 outline-none focus:border-gray-400 dark:focus:border-gray-600" />
          {searchText && (
            <button onClick={() => setSearchText('')} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500 hover:text-gray-700 dark:hover:text-gray-300">
              <X size={14} />
            </button>
          )}
        </div>
        <span className="text-[10px] text-gray-400 dark:text-gray-600 whitespace-nowrap">{filtered.length} logs</span>
      </div>

      <div ref={scrollRef} className="bg-gray-50/30 dark:bg-gray-900/30 border border-gray-200/60 dark:border-gray-800/60 rounded-xl overflow-auto font-mono text-xs" style={{ height: 'calc(100vh - 320px)' }}>
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 dark:text-gray-600">No logs to display</div>
        ) : (
          <table className="w-full">
            <tbody>
              {filtered.map(log => {
                const style = LEVEL_STYLES[log.level] || LEVEL_STYLES.info;
                const hasMetadata = log.metadata && Object.keys(log.metadata).length > 0;
                const expanded = expandedIds.has(log.id);
                return (
                  <React.Fragment key={log.id}>
                    <tr className="border-b border-gray-200/30 dark:border-gray-800/30 hover:bg-gray-500/[0.05] dark:hover:bg-white/[0.02] cursor-pointer" onClick={() => hasMetadata && toggleExpand(log.id)}>
                      <td className="px-3 py-1.5 text-gray-400 dark:text-gray-600 whitespace-nowrap w-[100px]">{formatTime(log.created_at)}</td>
                      <td className="px-2 py-1.5 w-[70px]">
                        <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${style.bg} ${style.text}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
                          {log.level}
                        </span>
                      </td>
                      <td className="px-2 py-1.5 w-[90px]">
                        <span className="text-gray-400 dark:text-gray-500 bg-gray-200/50 dark:bg-gray-800/50 px-1.5 py-0.5 rounded text-[10px]">{log.source}</span>
                      </td>
                      <td className="px-2 py-1.5 text-gray-700 dark:text-gray-300 break-all">{log.message}</td>
                    </tr>
                    {expanded && hasMetadata && (
                      <tr className="border-b border-gray-200/30 dark:border-gray-800/30">
                        <td colSpan={4} className="px-3 py-2 bg-gray-50/50 dark:bg-gray-900/50">
                          <pre className="text-[10px] text-gray-400 dark:text-gray-500 overflow-x-auto">{JSON.stringify(log.metadata, null, 2)}</pre>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
