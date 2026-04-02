import React, { useRef, useEffect, useMemo } from 'react';
import { Terminal, AlertTriangle, AlertCircle, Info } from 'lucide-react';
import type { EvalLogEntry } from '../../hooks/useEvalStream';

interface EvalLogTerminalProps {
  logs: EvalLogEntry[];
}

const MAX_VISIBLE_LOGS = 500;

export function EvalLogTerminal({ logs }: EvalLogTerminalProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Trim to max lines
  const visibleLogs = useMemo(
    () => (logs.length > MAX_VISIBLE_LOGS ? logs.slice(-MAX_VISIBLE_LOGS) : logs),
    [logs],
  );

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visibleLogs.length]);

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    return [
      d.getHours().toString().padStart(2, '0'),
      d.getMinutes().toString().padStart(2, '0'),
      d.getSeconds().toString().padStart(2, '0'),
    ].join(':');
  };

  const levelConfig: Record<string, { icon: typeof Info; color: string; timeColor: string }> = {
    info: {
      icon: Info,
      color: 'text-gray-500 dark:text-gray-400',
      timeColor: 'text-gray-400 dark:text-gray-600',
    },
    warn: {
      icon: AlertTriangle,
      color: 'text-yellow-400',
      timeColor: 'text-yellow-600/60',
    },
    error: {
      icon: AlertCircle,
      color: 'text-red-400',
      timeColor: 'text-red-600/60',
    },
  };

  return (
    <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-black rounded-lg overflow-hidden flex flex-col h-[280px]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-950 shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="h-3.5 w-3.5 text-green-400" />
          <span className="text-[11px] font-mono font-semibold uppercase tracking-widest text-gray-700 dark:text-gray-300">
            Logs
          </span>
        </div>
        <span className="text-[9px] font-mono text-gray-400 dark:text-gray-600">
          {logs.length} {logs.length > MAX_VISIBLE_LOGS ? `(showing ${MAX_VISIBLE_LOGS})` : ''} lines
        </span>
      </div>

      {/* Log output */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-2 space-y-0"
      >
        {visibleLogs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-[10px] font-mono text-gray-400 dark:text-gray-700">
              No logs yet...
            </span>
          </div>
        ) : (
          visibleLogs.map((log, i) => {
            const config = levelConfig[log.level] || levelConfig.info;
            const Icon = config.icon;

            return (
              <div
                key={i}
                className="flex items-start gap-2 py-0.5 px-1 hover:bg-gray-50 dark:hover:bg-gray-900/50 rounded transition-colors group"
              >
                <span className={`text-[10px] font-mono shrink-0 ${config.timeColor}`}>
                  {formatTime(log.timestamp)}
                </span>
                <Icon className={`h-3 w-3 shrink-0 mt-0.5 ${config.color}`} />
                <span className={`text-[11px] font-mono leading-relaxed break-all ${config.color}`}>
                  {log.message}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
