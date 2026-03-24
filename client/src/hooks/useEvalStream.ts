import { useState, useRef, useCallback, useEffect } from 'react';

export interface EvalFrame {
  grid: number[][];
  width: number;
  height: number;
  state: string;
  level: number;
  available_actions: string[];
}

export interface EvalTurnEntry {
  type: 'thinking' | 'action' | 'result' | 'message';
  turn: number;
  gameId: string;
  modelId: string;
  runIndex: number;
  action?: string;
  reasoning?: string;
  frame?: EvalFrame;
  score?: number;
  timestamp: number;
}

export interface EvalLogEntry {
  level: 'info' | 'error' | 'warn';
  message: string;
  timestamp: number;
}

export interface EvalRunResult {
  gameId: string;
  modelId: string;
  runIndex: number;
  score: number;
  state: string;
  totalTurns: number;
  totalTime: number;
}

export interface ModelRunState {
  modelId: string;
  gameId: string;
  runIndex: number;
  frame: EvalFrame | null;
  notepad: string;
  timeline: EvalTurnEntry[];
  lastAction: string | null;
  state: string; // NOT_FINISHED, WIN, GAME_OVER
}

export interface EvalState {
  status: 'idle' | 'connecting' | 'running' | 'completed' | 'cancelled' | 'error';
  evalId: string | null;
  // Per-model live state (key = modelId)
  modelStates: Record<string, ModelRunState>;
  // Current game being played
  currentGameId: string | null;
  // All completed run results
  results: EvalRunResult[];
  // Logs
  logs: EvalLogEntry[];
  // Errors
  error: string | null;
}

export function useEvalStream() {
  const [state, setState] = useState<EvalState>({
    status: 'idle', evalId: null,
    modelStates: {}, currentGameId: null,
    results: [], logs: [], error: null,
  });
  
  const eventSourceRef = useRef<EventSource | null>(null);
  
  const startEval = useCallback(async (config: {
    games: string[]; models: string[];
    runsPerGame: number; maxSteps: number; contextWindow: number;
    systemPrompt: string; reasoningEffort: string;
  }) => {
    // POST to /api/eval/start
    const res = await fetch('/api/eval/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('arc_token')}` },
      body: JSON.stringify(config),
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Failed to start eval'); }
    const { evalId } = await res.json();
    
    setState(prev => ({ ...prev, status: 'connecting', evalId, modelStates: {}, results: [], logs: [], error: null }));
    
    // Open SSE
    const es = new EventSource(`/api/eval/${evalId}/stream`);
    eventSourceRef.current = es;
    
    // Helper: update a specific model's state
    function updateModel(modelId: string, updater: (ms: ModelRunState) => ModelRunState) {
      setState(prev => ({
        ...prev,
        modelStates: {
          ...prev.modelStates,
          [modelId]: updater(prev.modelStates[modelId] || {
            modelId, gameId: '', runIndex: 0, frame: null, notepad: '', timeline: [], lastAction: null, state: 'NOT_FINISHED',
          }),
        },
      }));
    }
    
    es.addEventListener('eval.init', () => {
      setState(prev => ({ ...prev, status: 'running' }));
    });
    
    es.addEventListener('run.started', (e) => {
      const data = JSON.parse(e.data);
      setState(prev => ({
        ...prev,
        currentGameId: data.gameId,
        modelStates: {
          ...prev.modelStates,
          [data.modelId]: {
            modelId: data.modelId,
            gameId: data.gameId,
            runIndex: data.runIndex,
            frame: data.frame || null,
            notepad: '',
            timeline: [],
            lastAction: null,
            state: 'NOT_FINISHED',
          },
        },
      }));
    });
    
    es.addEventListener('turn.thinking', (e) => {
      const data = JSON.parse(e.data);
      updateModel(data.modelId, ms => ({
        ...ms,
        timeline: [...ms.timeline, {
          type: 'thinking', turn: data.turn, gameId: data.gameId,
          modelId: data.modelId, runIndex: data.runIndex, timestamp: Date.now(),
        }],
      }));
    });
    
    es.addEventListener('turn.action', (e) => {
      const data = JSON.parse(e.data);
      updateModel(data.modelId, ms => ({
        ...ms,
        lastAction: data.action,
        timeline: [...ms.timeline, {
          type: 'action', turn: data.turn, gameId: data.gameId,
          modelId: data.modelId, runIndex: data.runIndex,
          action: data.action, reasoning: data.reasoning, timestamp: Date.now(),
        }],
      }));
    });
    
    es.addEventListener('turn.result', (e) => {
      const data = JSON.parse(e.data);
      updateModel(data.modelId, ms => ({
        ...ms,
        frame: data.frame || ms.frame,
        state: data.frame?.state || ms.state,
        timeline: [...ms.timeline, {
          type: 'result', turn: data.turn, gameId: data.gameId,
          modelId: data.modelId, runIndex: data.runIndex,
          action: data.action, frame: data.frame, score: data.score, timestamp: Date.now(),
        }],
      }));
    });
    
    es.addEventListener('notepad.updated', (e) => {
      const data = JSON.parse(e.data);
      updateModel(data.modelId, ms => ({ ...ms, notepad: data.content || '' }));
    });
    
    es.addEventListener('run.completed', (e) => {
      const data = JSON.parse(e.data);
      setState(prev => ({ ...prev, results: [...prev.results, data] }));
    });
    
    es.addEventListener('log', (e) => {
      const data = JSON.parse(e.data);
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, { level: data.level, message: data.message, timestamp: Date.now() }].slice(-500),
      }));
    });
    
    es.addEventListener('eval.completed', () => {
      setState(prev => ({ ...prev, status: 'completed' }));
      es.close();
    });
    
    es.addEventListener('stream.complete', () => {
      setState(prev => ({ ...prev, status: prev.status === 'running' ? 'completed' : prev.status }));
      es.close();
    });
    
    es.addEventListener('stream.error', (e) => {
      const data = JSON.parse(e.data);
      setState(prev => ({ ...prev, status: 'error', error: data.message }));
      es.close();
    });
    
    es.onerror = () => {
      setState(prev => ({ ...prev, status: prev.status === 'running' ? 'error' : prev.status, error: 'Connection lost' }));
      es.close();
    };
    
    return evalId;
  }, []);
  
  const cancelEval = useCallback(async () => {
    if (!state.evalId) return;
    await fetch(`/api/eval/${state.evalId}/cancel`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${localStorage.getItem('arc_token')}` },
    });
    setState(prev => ({ ...prev, status: 'cancelled' }));
    eventSourceRef.current?.close();
  }, [state.evalId]);
  
  const reset = useCallback(() => {
    eventSourceRef.current?.close();
    setState({
      status: 'idle', evalId: null,
      modelStates: {}, currentGameId: null,
      results: [], logs: [], error: null,
    });
  }, []);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => { eventSourceRef.current?.close(); };
  }, []);
  
  return { ...state, startEval, cancelEval, reset };
}
