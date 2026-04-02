import React, { useState, useEffect } from 'react';
import {
  Settings, Play, XCircle, ChevronDown, ChevronUp,
  Cpu, Gamepad2, Hash, Layers, Brain, FileText,
} from 'lucide-react';
import { gamesAPI } from '../../api/client';

interface GameOption {
  game_id: string;
  name: string;
}

interface ModelOption {
  id: string;
  name: string;
  color: string;
}

interface EvalConfig {
  games: string[];
  models: string[];
  runsPerGame: number;
  maxSteps: number;
  contextWindow: number;
  reasoningEffort: string;
  systemPrompt: string;
}

interface EvalConfigPanelProps {
  onStart: (config: EvalConfig) => void;
  onCancel: () => void;
  isRunning: boolean;
}

export function EvalConfigPanel({ onStart, onCancel, isRunning }: EvalConfigPanelProps) {
  const [games, setGames] = useState<GameOption[]>([]);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [gamesLoading, setGamesLoading] = useState(true);
  const [modelsLoading, setModelsLoading] = useState(true);

  const [selectedGames, setSelectedGames] = useState<Set<string>>(new Set());
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [runsPerGame, setRunsPerGame] = useState(3);
  const [maxSteps, setMaxSteps] = useState(200);
  const [contextWindow, setContextWindow] = useState(10);
  const [reasoningEffort, setReasoningEffort] = useState('medium');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [promptExpanded, setPromptExpanded] = useState(false);

  const [gamesExpanded, setGamesExpanded] = useState(true);
  const [modelsExpanded, setModelsExpanded] = useState(true);

  // Fetch games
  useEffect(() => {
    gamesAPI.list()
      .then((res) => {
        const list = (res.data?.games || res.data || []).map((g: any) => ({
          game_id: g.game_id || g.gameId,
          name: g.name || g.game_id || g.gameId,
        }));
        setGames(list);
      })
      .catch(() => setGames([]))
      .finally(() => setGamesLoading(false));
  }, []);

  // Fetch models
  useEffect(() => {
    fetch('/api/eval/models', {
      headers: { Authorization: `Bearer ${localStorage.getItem('arc_token')}` },
    })
      .then((r) => r.json())
      .then((data) => {
        const list = (data.models || data || []).map((m: any) => ({
          id: m.id || m.key,
          name: m.name || m.id || m.key,
          color: m.color || 'gray',
        }));
        setModels(list);
      })
      .catch(() => setModels([]))
      .finally(() => setModelsLoading(false));
  }, []);

  const toggleGame = (id: string) => {
    setSelectedGames((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const toggleModel = (id: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const canStart = selectedGames.size > 0 && selectedModels.size > 0 && !isRunning;

  const handleStart = () => {
    onStart({
      games: Array.from(selectedGames),
      models: Array.from(selectedModels),
      runsPerGame,
      maxSteps,
      contextWindow,
      reasoningEffort,
      systemPrompt,
    });
  };

  const MODEL_COLORS: Record<string, string> = {
    blue: 'bg-blue-400',
    green: 'bg-green-400',
    amber: 'bg-amber-400',
    purple: 'bg-purple-400',
    red: 'bg-red-400',
    cyan: 'bg-cyan-400',
    pink: 'bg-pink-400',
    gray: 'bg-gray-400',
  };

  return (
    <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-gray-200 dark:border-gray-800">
        <Settings className="h-3.5 w-3.5 text-blue-400" />
        <span className="text-[11px] font-mono font-semibold uppercase tracking-widest text-gray-700 dark:text-gray-300">
          Eval Config
        </span>
      </div>

      <div className="p-3 space-y-3">
        {/* Game Selector */}
        <div className="space-y-1.5">
          <button
            onClick={() => setGamesExpanded(!gamesExpanded)}
            className="flex items-center justify-between w-full text-left"
          >
            <div className="flex items-center gap-1.5">
              <Gamepad2 className="h-3 w-3 text-emerald-400" />
              <span className="text-[10px] font-mono uppercase tracking-widest text-gray-500 dark:text-gray-400">
                Games
              </span>
              {selectedGames.size > 0 && (
                <span className="text-[9px] font-mono text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">
                  {selectedGames.size}
                </span>
              )}
            </div>
            {gamesExpanded ? (
              <ChevronUp className="h-3 w-3 text-gray-400 dark:text-gray-600" />
            ) : (
              <ChevronDown className="h-3 w-3 text-gray-400 dark:text-gray-600" />
            )}
          </button>

          {gamesExpanded && (
            <div className="max-h-40 overflow-y-auto space-y-0.5 border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-950 p-1.5">
              {gamesLoading ? (
                <div className="text-[10px] text-gray-400 dark:text-gray-600 font-mono py-2 text-center">Loading games...</div>
              ) : games.length === 0 ? (
                <div className="text-[10px] text-gray-400 dark:text-gray-600 font-mono py-2 text-center">No games found</div>
              ) : (
                games.map((g) => (
                  <label
                    key={g.game_id}
                    className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer transition-colors text-[11px] font-mono ${
                      selectedGames.has(g.game_id)
                        ? 'bg-emerald-500/10 text-emerald-300'
                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedGames.has(g.game_id)}
                      onChange={() => toggleGame(g.game_id)}
                      className="h-3 w-3 rounded border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 text-emerald-500 focus:ring-0 focus:ring-offset-0"
                    />
                    <span className="truncate flex-1">{g.name}</span>
                    <span className="text-[9px] text-gray-400 dark:text-gray-600 shrink-0">{g.game_id}</span>
                  </label>
                ))
              )}
            </div>
          )}
        </div>

        {/* Model Selector */}
        <div className="space-y-1.5">
          <button
            onClick={() => setModelsExpanded(!modelsExpanded)}
            className="flex items-center justify-between w-full text-left"
          >
            <div className="flex items-center gap-1.5">
              <Cpu className="h-3 w-3 text-purple-400" />
              <span className="text-[10px] font-mono uppercase tracking-widest text-gray-500 dark:text-gray-400">
                Models
              </span>
              {selectedModels.size > 0 && (
                <span className="text-[9px] font-mono text-purple-400 bg-purple-400/10 px-1.5 py-0.5 rounded">
                  {selectedModels.size}
                </span>
              )}
            </div>
            {modelsExpanded ? (
              <ChevronUp className="h-3 w-3 text-gray-400 dark:text-gray-600" />
            ) : (
              <ChevronDown className="h-3 w-3 text-gray-400 dark:text-gray-600" />
            )}
          </button>

          {modelsExpanded && (
            <div className="max-h-36 overflow-y-auto space-y-0.5 border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-950 p-1.5">
              {modelsLoading ? (
                <div className="text-[10px] text-gray-400 dark:text-gray-600 font-mono py-2 text-center">Loading models...</div>
              ) : models.length === 0 ? (
                <div className="text-[10px] text-gray-400 dark:text-gray-600 font-mono py-2 text-center">No models found</div>
              ) : (
                models.map((m) => (
                  <label
                    key={m.id}
                    className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer transition-colors text-[11px] font-mono ${
                      selectedModels.has(m.id)
                        ? 'bg-purple-500/10 text-purple-300'
                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedModels.has(m.id)}
                      onChange={() => toggleModel(m.id)}
                      className="h-3 w-3 rounded border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-gray-800 text-purple-500 focus:ring-0 focus:ring-offset-0"
                    />
                    <span className={`h-2 w-2 rounded-full shrink-0 ${MODEL_COLORS[m.color] || 'bg-gray-400'}`} />
                    <span className="truncate">{m.name}</span>
                  </label>
                ))
              )}
            </div>
          )}
        </div>

        {/* Numeric Configs */}
        <div className="grid grid-cols-3 gap-2">
          <div className="space-y-1">
            <label className="flex items-center gap-1 text-[9px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500">
              <Hash className="h-2.5 w-2.5" />
              Runs
            </label>
            <input
              type="number"
              min={1}
              max={10}
              value={runsPerGame}
              onChange={(e) => setRunsPerGame(Math.min(10, Math.max(1, Number(e.target.value))))}
              disabled={isRunning}
              className="w-full bg-gray-50 dark:bg-gray-950 border border-gray-300 dark:border-gray-700 text-gray-800 dark:text-gray-200 text-xs font-mono px-2 py-1.5 rounded focus:outline-none focus:border-blue-500/60 transition-colors disabled:opacity-50"
            />
          </div>
          <div className="space-y-1">
            <label className="flex items-center gap-1 text-[9px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500">
              <Layers className="h-2.5 w-2.5" />
              Max Steps
            </label>
            <input
              type="number"
              min={50}
              max={500}
              step={10}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Math.min(500, Math.max(50, Number(e.target.value))))}
              disabled={isRunning}
              className="w-full bg-gray-50 dark:bg-gray-950 border border-gray-300 dark:border-gray-700 text-gray-800 dark:text-gray-200 text-xs font-mono px-2 py-1.5 rounded focus:outline-none focus:border-blue-500/60 transition-colors disabled:opacity-50"
            />
          </div>
          <div className="space-y-1">
            <label className="flex items-center gap-1 text-[9px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500">
              <Brain className="h-2.5 w-2.5" />
              Context
            </label>
            <input
              type="number"
              min={5}
              max={20}
              value={contextWindow}
              onChange={(e) => setContextWindow(Math.min(20, Math.max(5, Number(e.target.value))))}
              disabled={isRunning}
              className="w-full bg-gray-50 dark:bg-gray-950 border border-gray-300 dark:border-gray-700 text-gray-800 dark:text-gray-200 text-xs font-mono px-2 py-1.5 rounded focus:outline-none focus:border-blue-500/60 transition-colors disabled:opacity-50"
            />
          </div>
        </div>

        {/* Reasoning Effort */}
        <div className="space-y-1">
          <label className="text-[9px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500">
            Reasoning Effort
          </label>
          <select
            value={reasoningEffort}
            onChange={(e) => setReasoningEffort(e.target.value)}
            disabled={isRunning}
            className="w-full bg-gray-50 dark:bg-gray-950 border border-gray-300 dark:border-gray-700 text-gray-800 dark:text-gray-200 text-xs font-mono px-2 py-1.5 rounded focus:outline-none focus:border-blue-500/60 transition-colors disabled:opacity-50"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        {/* System Prompt */}
        <div className="space-y-1">
          <button
            onClick={() => setPromptExpanded(!promptExpanded)}
            className="flex items-center justify-between w-full text-left"
          >
            <div className="flex items-center gap-1">
              <FileText className="h-2.5 w-2.5 text-gray-400 dark:text-gray-500" />
              <span className="text-[9px] font-mono uppercase tracking-wider text-gray-400 dark:text-gray-500">
                System Prompt
              </span>
            </div>
            {promptExpanded ? (
              <ChevronUp className="h-3 w-3 text-gray-400 dark:text-gray-600" />
            ) : (
              <ChevronDown className="h-3 w-3 text-gray-400 dark:text-gray-600" />
            )}
          </button>
          {promptExpanded && (
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              disabled={isRunning}
              placeholder="Enter system prompt (optional, uses playbook default if empty)..."
              rows={6}
              className="w-full bg-gray-50 dark:bg-gray-950 border border-gray-300 dark:border-gray-700 text-gray-800 dark:text-gray-200 text-[11px] font-mono px-2 py-1.5 rounded resize-y focus:outline-none focus:border-blue-500/60 transition-colors disabled:opacity-50 placeholder:text-gray-400 dark:placeholder:text-gray-700"
            />
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2 pt-1">
          <button
            onClick={handleStart}
            disabled={!canStart}
            className="flex-1 flex items-center justify-center gap-2 py-2 text-[11px] font-mono font-semibold uppercase tracking-widest rounded transition-all
              bg-emerald-600 hover:bg-emerald-500 text-white
              disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:text-gray-400 dark:disabled:text-gray-600 disabled:cursor-not-allowed"
          >
            <Play className="h-3.5 w-3.5" />
            Start Eval
          </button>

          {isRunning && (
            <button
              onClick={onCancel}
              className="flex items-center justify-center gap-2 px-4 py-2 text-[11px] font-mono font-semibold uppercase tracking-widest rounded transition-all
                bg-red-600/20 hover:bg-red-600/40 text-red-400 border border-red-500/30 hover:border-red-500/60"
            >
              <XCircle className="h-3.5 w-3.5" />
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
