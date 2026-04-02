import React, { useMemo } from 'react';
import { BarChart2, Trophy, Skull, Clock, Hash } from 'lucide-react';
import type { EvalRunResult } from '../../hooks/useEvalStream';

interface EvalResultsPanelProps {
  results: EvalRunResult[];
}

export function EvalResultsPanel({ results }: EvalResultsPanelProps) {
  const stats = useMemo(() => {
    if (results.length === 0) return null;
    const wins = results.filter((r) => r.state === 'WIN').length;
    const avgTurns = results.reduce((sum, r) => sum + r.totalTurns, 0) / results.length;
    const avgTime = results.reduce((sum, r) => sum + r.totalTime, 0) / results.length;
    return { total: results.length, wins, avgTurns: Math.round(avgTurns), avgTime };
  }, [results]);

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
  };

  const stateBadge = (state: string) => {
    if (state === 'WIN') {
      return (
        <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold uppercase px-1.5 py-0.5 rounded bg-emerald-400/10 text-emerald-400 border border-emerald-500/20">
          <Trophy className="h-2.5 w-2.5" />
          Win
        </span>
      );
    }
    if (state === 'GAME_OVER' || state === 'LOSE') {
      return (
        <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold uppercase px-1.5 py-0.5 rounded bg-red-400/10 text-red-400 border border-red-500/20">
          <Skull className="h-2.5 w-2.5" />
          Game Over
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 text-[9px] font-mono font-semibold uppercase px-1.5 py-0.5 rounded bg-gray-200/30 dark:bg-gray-700/30 text-gray-500 dark:text-gray-400 border border-gray-300/50 dark:border-gray-700/50">
        Not Finished
      </span>
    );
  };

  return (
    <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-gray-200 dark:border-gray-800">
        <BarChart2 className="h-3.5 w-3.5 text-amber-400" />
        <span className="text-[11px] font-mono font-semibold uppercase tracking-widest text-gray-700 dark:text-gray-300">
          Results
        </span>
        {results.length > 0 && (
          <span className="text-[9px] font-mono text-gray-400 dark:text-gray-600">
            {results.length} runs
          </span>
        )}
      </div>

      {/* Summary Stats */}
      {stats && (
        <div className="grid grid-cols-3 gap-px bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-800">
          <div className="bg-white dark:bg-gray-900 px-3 py-2 text-center">
            <p className="text-[16px] font-mono font-bold text-gray-900 dark:text-white">{stats.total}</p>
            <p className="text-[8px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500 mt-0.5">
              Total Runs
            </p>
          </div>
          <div className="bg-white dark:bg-gray-900 px-3 py-2 text-center">
            <p className="text-[16px] font-mono font-bold text-emerald-400">
              {stats.wins}
              <span className="text-[10px] text-gray-400 dark:text-gray-600 font-normal">
                /{stats.total}
              </span>
            </p>
            <p className="text-[8px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500 mt-0.5">
              Wins
            </p>
          </div>
          <div className="bg-white dark:bg-gray-900 px-3 py-2 text-center">
            <p className="text-[16px] font-mono font-bold text-blue-400">{stats.avgTurns}</p>
            <p className="text-[8px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500 mt-0.5">
              Avg Turns
            </p>
          </div>
        </div>
      )}

      {/* Results List */}
      <div className="max-h-[400px] overflow-y-auto">
        {results.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <span className="text-[10px] font-mono text-gray-400 dark:text-gray-600">
              No completed runs yet
            </span>
          </div>
        ) : (
          <div className="divide-y divide-gray-200/50 dark:divide-gray-800/50">
            {results.map((r, i) => (
              <div
                key={i}
                className={`flex items-center gap-2 px-3 py-2 text-[11px] font-mono transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/30 ${
                  r.state === 'WIN' ? 'border-l-2 border-l-emerald-500/40' : 'border-l-2 border-l-transparent'
                }`}
              >
                {/* Game + Model */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-gray-700 dark:text-gray-300 truncate">{r.gameId}</span>
                    <span className="text-gray-300 dark:text-gray-700">|</span>
                    <span className="text-purple-600 dark:text-purple-300 truncate">{r.modelId}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[9px] text-gray-400 dark:text-gray-600">
                      <Hash className="h-2 w-2 inline" /> Run {r.runIndex + 1}
                    </span>
                  </div>
                </div>

                {/* Score */}
                <div className="text-right shrink-0 w-10">
                  <span className={`font-semibold ${r.state === 'WIN' ? 'text-emerald-400' : 'text-gray-500 dark:text-gray-400'}`}>
                    {r.score}
                  </span>
                </div>

                {/* State Badge */}
                <div className="shrink-0">{stateBadge(r.state)}</div>

                {/* Turns */}
                <div className="shrink-0 w-14 text-right">
                  <div className="flex items-center justify-end gap-1 text-gray-400 dark:text-gray-500">
                    <Hash className="h-2.5 w-2.5" />
                    <span>{r.totalTurns}</span>
                  </div>
                </div>

                {/* Time */}
                <div className="shrink-0 w-16 text-right">
                  <div className="flex items-center justify-end gap-1 text-gray-400 dark:text-gray-500">
                    <Clock className="h-2.5 w-2.5" />
                    <span>{formatTime(r.totalTime)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
