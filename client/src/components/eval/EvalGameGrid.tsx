import React from 'react';
import { Monitor, Layers, Activity } from 'lucide-react';
import GameCanvas from '../GameCanvas';
import type { EvalFrame } from '../../hooks/useEvalStream';

interface EvalGameGridProps {
  frame: EvalFrame | null;
  gameId: string | null;
  modelId: string | null;
  runIndex: number;
  compact?: boolean;
}

export function EvalGameGrid({ frame, gameId, modelId, runIndex, compact = false }: EvalGameGridProps) {
  if (!frame) {
    return (
      <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 rounded-lg overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-gray-200 dark:border-gray-800">
          <Monitor className="h-3.5 w-3.5 text-blue-400" />
          <span className="text-[11px] font-mono font-semibold uppercase tracking-widest text-gray-700 dark:text-gray-300">
            Game Grid
          </span>
        </div>
        <div className="flex items-center justify-center py-20">
          <span className="text-[11px] font-mono text-gray-400 dark:text-gray-600">
            Waiting for eval to start...
          </span>
        </div>
      </div>
    );
  }

  const stateColor =
    frame.state === 'WIN'
      ? 'text-emerald-400'
      : frame.state === 'GAME_OVER' || frame.state === 'LOSE'
        ? 'text-red-400'
        : 'text-amber-400';

  const stateBg =
    frame.state === 'WIN'
      ? 'bg-emerald-400/10 border-emerald-500/30'
      : frame.state === 'GAME_OVER' || frame.state === 'LOSE'
        ? 'bg-red-400/10 border-red-500/30'
        : 'bg-amber-400/10 border-amber-500/30';

  return (
    <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-2">
          <Monitor className="h-3.5 w-3.5 text-blue-400" />
          <span className="text-[11px] font-mono text-gray-800 dark:text-gray-200">
            {gameId || '—'}
          </span>
          <span className="text-gray-300 dark:text-gray-700">|</span>
          <span className="text-[11px] font-mono text-purple-600 dark:text-purple-300">
            {modelId || '—'}
          </span>
          <span className="text-gray-300 dark:text-gray-700">|</span>
          <span className="text-[11px] font-mono text-gray-500 dark:text-gray-400">
            Run {runIndex + 1}
          </span>
        </div>
        <span className={`text-[10px] font-mono font-semibold uppercase px-2 py-0.5 rounded border ${stateBg} ${stateColor}`}>
          {frame.state === 'NOT_FINISHED' ? 'Playing' : frame.state}
        </span>
      </div>

      {/* Grid */}
      <div className={`flex items-center justify-center ${compact ? 'p-2' : 'p-4'} bg-gray-50 dark:bg-gray-950`}>
        <GameCanvas
          grid={frame.grid}
          width={frame.width}
          height={frame.height}
          interactive={false}
          maxCanvasWidth={compact ? 240 : 480}
          maxCanvasHeight={compact ? 240 : 480}
          showGrid={!compact}
        />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-3 py-2 border-t border-gray-200 dark:border-gray-800 text-[10px] font-mono">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-gray-400 dark:text-gray-500">
            <Layers className="h-2.5 w-2.5" />
            <span>Level: {frame.level}</span>
          </div>
          <div className="flex items-center gap-1 text-gray-400 dark:text-gray-500">
            <Activity className="h-2.5 w-2.5" />
            <span className={stateColor}>{frame.state}</span>
          </div>
        </div>
        {frame.available_actions && frame.available_actions.length > 0 && (
          <div className="text-gray-400 dark:text-gray-600">
            Actions: [{frame.available_actions.join(', ')}]
          </div>
        )}
      </div>
    </div>
  );
}
