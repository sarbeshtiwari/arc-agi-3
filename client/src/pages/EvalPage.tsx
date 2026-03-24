import React, { useCallback, useState } from 'react';
import { Loader2, FlaskConical, AlertTriangle, Eye } from 'lucide-react';
import { useEvalStream } from '../hooks/useEvalStream';
import { EvalConfigPanel } from '../components/eval/EvalConfigPanel';
import { EvalResultsPanel } from '../components/eval/EvalResultsPanel';
import { EvalGameGrid } from '../components/eval/EvalGameGrid';
import { EvalReasoningPanel } from '../components/eval/EvalReasoningPanel';
import { EvalNotepad } from '../components/eval/EvalNotepad';
import { EvalLogTerminal } from '../components/eval/EvalLogTerminal';

// Model colors for visual distinction
const MODEL_COLORS: Record<string, string> = {
  'chatgpt': '#10B981',
  'gemini': '#3B82F6',
  'kimi-k2.5': '#8B5CF6',
  'claude-opus-4.6': '#D946EF',
};

export default function EvalPage() {
  const {
    status, evalId, modelStates, currentGameId,
    results, logs, error,
    startEval, cancelEval, reset,
  } = useEvalStream();

  // Which model's reasoning/notepad to show in the right panel
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const isRunning = status === 'running' || status === 'connecting';
  const activeModels = Object.values(modelStates);
  const displayModel = selectedModel && modelStates[selectedModel] ? modelStates[selectedModel] : activeModels[0] || null;

  const handleStart = useCallback(async (config: any) => {
    try { await startEval(config); } catch (err: any) { console.error('[EvalPage]', err); }
  }, [startEval]);

  const handleCancel = useCallback(async () => {
    try { await cancelEval(); } catch (err: any) { console.error('[EvalPage]', err); }
  }, [cancelEval]);

  return (
    <div className="min-h-[calc(100vh-48px)]">
      {/* Page header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center">
            <FlaskConical className="h-4.5 w-4.5 text-blue-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Eval Runner</h1>
            <p className="text-xs text-gray-500">Run models against games — all models play in parallel</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-mono uppercase tracking-widest px-2.5 py-1 border rounded ${
            status === 'running' ? 'border-emerald-700 bg-emerald-950/30 text-emerald-400' :
            status === 'connecting' ? 'border-amber-700 bg-amber-950/30 text-amber-400 animate-pulse' :
            status === 'error' ? 'border-red-700 bg-red-950/30 text-red-400' :
            status === 'completed' ? 'border-blue-700 bg-blue-950/30 text-blue-400' :
            'border-gray-700 bg-gray-900 text-gray-500'
          }`}>{status}</span>
          {(status === 'completed' || status === 'cancelled' || status === 'error') && (
            <button onClick={reset} className="text-[10px] font-mono text-gray-400 hover:text-white px-2 py-1 border border-gray-700 rounded hover:border-gray-500 transition-colors">Reset</button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 flex items-center gap-2 px-4 py-2.5 bg-red-500/10 border border-red-500/30 rounded-lg">
          <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
          <span className="text-sm text-red-300 font-mono">{error}</span>
        </div>
      )}

      {/* Layout: Config (left) | Model Grids (center) | Reasoning (right) */}
      <div className="flex gap-4 items-start">
        {/* Left: Config + Results */}
        <div className="w-72 shrink-0 space-y-3">
          <EvalConfigPanel onStart={handleStart} onCancel={handleCancel} isRunning={isRunning} />
          <EvalResultsPanel results={results} />
        </div>

        {/* Center: Game Grids — one per active model */}
        <div className="flex-1 min-w-0 space-y-3">
          {isRunning && currentGameId && (
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 border border-gray-800 rounded-lg">
              <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin" />
              <span className="text-[11px] font-mono text-gray-300">Game:</span>
              <span className="text-[11px] font-mono text-emerald-300 font-semibold">{currentGameId}</span>
              <span className="text-gray-700">|</span>
              <span className="text-[11px] font-mono text-gray-400">{activeModels.length} models playing</span>
            </div>
          )}

          {/* Model grid cards */}
          {activeModels.length > 0 ? (
            <div className={`grid gap-3 ${
              activeModels.length === 1 ? 'grid-cols-1' :
              activeModels.length === 2 ? 'grid-cols-2' :
              activeModels.length <= 4 ? 'grid-cols-2' : 'grid-cols-3'
            }`}>
              {activeModels.map((ms) => {
                const color = MODEL_COLORS[ms.modelId] || '#6B7280';
                const isSelected = selectedModel === ms.modelId;
                const turnCount = ms.timeline.filter(t => t.type === 'action').length;
                return (
                  <div
                    key={ms.modelId}
                    onClick={() => setSelectedModel(ms.modelId)}
                    className={`bg-gray-900 rounded-xl border p-3 cursor-pointer transition-all ${
                      isSelected ? 'border-blue-500/50 ring-1 ring-blue-500/20' : 'border-gray-800 hover:border-gray-700'
                    }`}
                  >
                    {/* Model header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                        <span className="text-xs font-bold text-white">{ms.modelId}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                          ms.state === 'WIN' ? 'bg-green-500/20 text-green-400' :
                          ms.state === 'GAME_OVER' ? 'bg-red-500/20 text-red-400' :
                          'bg-blue-500/10 text-blue-400'
                        }`}>{ms.state}</span>
                        {isSelected && <Eye size={10} className="text-blue-400" />}
                      </div>
                    </div>

                    {/* Game grid */}
                    <EvalGameGrid
                      frame={ms.frame}
                      gameId={ms.gameId}
                      modelId={ms.modelId}
                      runIndex={ms.runIndex}
                      compact
                    />

                    {/* Stats footer */}
                    <div className="flex items-center justify-between mt-2 text-[9px] font-mono text-gray-500">
                      <span>Run {ms.runIndex + 1} | Lv {ms.frame?.level ?? 0}</span>
                      <span>{turnCount} turns</span>
                      {ms.lastAction && (
                        <span className="text-gray-400">{ms.lastAction}</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-12 text-center">
              <FlaskConical className="mx-auto h-10 w-10 text-gray-700 mb-3" />
              <p className="text-gray-500 text-sm">Select games and models, then click Start</p>
            </div>
          )}

          {/* Log terminal below grids */}
          <EvalLogTerminal logs={logs} />
        </div>

        {/* Right: Reasoning + Notepad for selected model */}
        <div className="w-80 shrink-0 space-y-3">
          {displayModel ? (
            <>
              <div className="flex items-center gap-2 px-3 py-2 bg-gray-900 border border-gray-800 rounded-lg">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: MODEL_COLORS[displayModel.modelId] || '#6B7280' }} />
                <span className="text-[11px] font-mono text-white font-bold">{displayModel.modelId}</span>
                <span className="text-[9px] text-gray-500">click a grid to switch</span>
              </div>
              <EvalReasoningPanel timeline={displayModel.timeline} />
              <EvalNotepad
                content={displayModel.notepad}
                gameId={displayModel.gameId}
                modelId={displayModel.modelId}
                runIndex={displayModel.runIndex}
              />
            </>
          ) : (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-8 text-center">
              <p className="text-gray-600 text-xs">Reasoning panel appears when models are running</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
