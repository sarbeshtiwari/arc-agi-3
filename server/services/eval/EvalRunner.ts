/*
  EvalRunner — core evaluation orchestrator.

  Runs LLM models against ARC-AGI games and emits real-time SSE events.
  For each game × model × run, it spawns a GamePythonBridge, builds a
  conversational agent loop (system prompt → turn message → LLM call →
  action extraction → game step), and streams every step through SSEStreamManager.

  The agent uses a persistent notepad (<NOTEPAD>...</NOTEPAD>) across turns
  and a sliding context window to keep conversations within token limits.

  Dependencies:
    - GamePythonBridge (Python subprocess for game execution)
    - callLLM / AVAILABLE_MODELS (multi-provider LLM abstraction)
    - sseManager (SSE event delivery to connected clients)
    - queryOne (Postgres helper, available for future result persistence)
*/

import * as path from 'path';
import { GamePythonBridge, type GameFrame } from '../GamePythonBridge.js';
import { callLLM, AVAILABLE_MODELS, type LLMMessage } from './LLMProvider.js';
import { sseManager } from './SSEStreamManager.js';
import { queryOne } from '../../db.js';

// ── Public types ──────────────────────────────────────────────────────────

export interface EvalConfig {
  evalId: string;
  games: Array<{ gameId: string; gameCode: string; localDir: string }>;
  models: string[];        // model IDs from AVAILABLE_MODELS
  runsPerGame: number;     // 1-10
  maxSteps: number;        // 50-500
  contextWindow: number;   // 5-20 turns of history
  systemPrompt: string;
  reasoningEffort: 'low' | 'medium' | 'high';
}

export interface EvalSession {
  evalId: string;
  config: EvalConfig;
  status: 'pending' | 'running' | 'completed' | 'cancelled' | 'error';
  startedAt: number;
  results: EvalRunResult[];
}

export interface EvalRunResult {
  gameId: string;
  modelId: string;
  runIndex: number;
  score: number;
  state: string;
  totalTurns: number;
  totalTime: number;
  actions: string[];
}

// ── In-memory storage ─────────────────────────────────────────────────────

const evalSessions = new Map<string, EvalSession>();

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Retrieve an eval session by ID. Returns undefined if not found.
 */
export function getEvalSession(evalId: string): EvalSession | undefined {
  return evalSessions.get(evalId);
}

export function getAllSessions(): EvalSession[] {
  return Array.from(evalSessions.values()).sort((a, b) => b.startedAt - a.startedAt);
}

/**
 * Cancel a running eval. The agent loops check this flag between turns
 * and will bail out gracefully.
 */
export function cancelEval(evalId: string): void {
  const session = evalSessions.get(evalId);
  if (session && session.status === 'running') {
    session.status = 'cancelled';
    sseManager.sendEvent(evalId, 'eval.cancelled', { evalId });
    sseManager.sendEvent(evalId, 'log', {
      level: 'warn',
      message: `Eval ${evalId} cancelled by user`,
    });
  }
}

/**
 * Start an evaluation run. Creates a session, then iterates sequentially
 * over every game × model × run combination, running the agent loop for
 * each. SSE events are emitted throughout for real-time UI updates.
 */
export async function startEval(config: EvalConfig): Promise<void> {
  const session: EvalSession = {
    evalId: config.evalId,
    config,
    status: 'running',
    startedAt: Date.now(),
    results: [],
  };
  evalSessions.set(config.evalId, session);

  sseManager.sendEvent(config.evalId, 'eval.init', {
    evalId: config.evalId,
    games: config.games.map((g) => g.gameId),
    models: config.models,
    config,
  });

  sseManager.sendEvent(config.evalId, 'log', {
    level: 'info',
    message: `Eval started: ${config.games.length} games × ${config.models.length} models × ${config.runsPerGame} runs = ${config.games.length * config.models.length * config.runsPerGame} total runs`,
  });

  // For each game, run ALL models in PARALLEL (each model gets its own game bridge)
  for (const game of config.games) {
    if (session.status === 'cancelled') break;

    sseManager.sendEvent(config.evalId, 'log', {
      level: 'info',
      message: `── Game: ${game.gameId} ── Running ${config.models.length} models in parallel`,
    });

    // Build parallel tasks: all models × all runs for this game
    const parallelTasks: Array<{ modelId: string; runIndex: number }> = [];
    for (const modelId of config.models) {
      for (let runIndex = 0; runIndex < config.runsPerGame; runIndex++) {
        parallelTasks.push({ modelId, runIndex });
      }
    }

    // Run all in parallel (each model is stateless, each gets its own Python bridge)
    const taskPromises = parallelTasks.map(async ({ modelId, runIndex }) => {
      if (session.status === 'cancelled') return;
      try {
        await runSingleGame(config, session, game, modelId, runIndex);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        sseManager.sendEvent(config.evalId, 'log', {
          level: 'error',
          message: `[${game.gameId}] [${modelId}] Run ${runIndex + 1} failed: ${msg}`,
        });
        session.results.push({
          gameId: game.gameId,
          modelId,
          runIndex,
          score: 0,
          state: 'ERROR',
          totalTurns: 0,
          totalTime: 0,
          actions: [],
        });
      }
    });

    await Promise.all(taskPromises);
  }

  // Finalise
  session.status = session.status === 'cancelled' ? 'cancelled' : 'completed';

  sseManager.sendEvent(config.evalId, 'eval.completed', {
    status: session.status,
    summary: {
      results: session.results,
      totalRuns: session.results.length,
      completedAt: Date.now(),
      elapsedMs: Date.now() - session.startedAt,
    },
  });

  sseManager.close(config.evalId);
}

// ── Core agent loop ───────────────────────────────────────────────────────

type GameInfo = EvalConfig['games'][number];

async function runSingleGame(
  config: EvalConfig,
  session: EvalSession,
  game: GameInfo,
  modelId: string,
  runIndex: number,
): Promise<void> {
  const bridge = new GamePythonBridge();
  const gamePyPath = path.join(game.localDir, `${game.gameCode}.py`);

  try {
    // ── Init game ───────────────────────────────────────────────────
    const initFrame = await bridge.init(game.gameId, gamePyPath, runIndex);

    sseManager.sendEvent(config.evalId, 'run.started', {
      gameId: game.gameId,
      modelId,
      runIndex,
      frame: initFrame,
    });

    sseManager.sendEvent(config.evalId, 'log', {
      level: 'info',
      message: `[${game.gameId}] [${modelId}] Run ${runIndex + 1}/${config.runsPerGame} started`,
    });

    // ── Build conversation ──────────────────────────────────────────
    const systemMessage = buildSystemPrompt(config.systemPrompt, config.reasoningEffort);
    const messages: LLMMessage[] = [
      { role: 'system', content: systemMessage },
    ];

    let frame: GameFrame = initFrame;
    let notepad = '';
    const startTime = Date.now();
    const actions: string[] = [];

    // ── Turn loop ───────────────────────────────────────────────────
    for (let turn = 0; turn < config.maxSteps; turn++) {
      // Cancellation guard
      if (session.status === 'cancelled') break;

      // Terminal state guard
      if (frame.state === 'WIN' || frame.state === 'GAME_OVER') break;

      // Build turn message with current game state
      const turnMessage = buildTurnMessage(frame, turn, notepad, config.maxSteps);
      messages.push({ role: 'user', content: turnMessage });

      // Trim conversation to stay within context window
      trimHistory(messages, config.contextWindow);

      // Log turn start
      sseManager.sendEvent(config.evalId, 'log', {
        level: 'info',
        message: `[${game.gameId}] [${modelId}] Turn ${turn + 1}/${config.maxSteps}`,
      });

      // Notify UI that the model is thinking
      sseManager.sendEvent(config.evalId, 'turn.thinking', {
        gameId: game.gameId,
        modelId,
        runIndex,
        turn,
      });

      // ── Call LLM ────────────────────────────────────────────────
      let llmResponse: string;
      try {
        const result = await callLLM(modelId, messages, {
          maxTokens: 1024,
          temperature: 0.3,
        });
        llmResponse = result.content;
        sseManager.sendEvent(config.evalId, 'log', {
          level: 'info',
          message: `[${game.gameId}] [${modelId}] Tokens: ${result.inputTokens}→${result.outputTokens}`,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        sseManager.sendEvent(config.evalId, 'log', {
          level: 'error',
          message: `[${game.gameId}] [${modelId}] LLM error: ${msg}`,
        });
        break;
      }

      // Record assistant response in conversation
      messages.push({ role: 'assistant', content: llmResponse });

      // ── Extract notepad ───────────────────────────────────────────
      const newNotepad = extractNotepad(llmResponse);
      if (newNotepad !== null) {
        notepad = newNotepad;
        sseManager.sendEvent(config.evalId, 'notepad.updated', {
          gameId: game.gameId,
          modelId,
          runIndex,
          content: notepad,
        });
      }

      // ── Extract action ────────────────────────────────────────────
      const availableActions = (frame.available_actions ?? []).map(String);
      const action = extractAction(llmResponse, availableActions);
      actions.push(action);

      // Send reasoning + action event
      sseManager.sendEvent(config.evalId, 'turn.action', {
        gameId: game.gameId,
        modelId,
        runIndex,
        turn,
        action,
        reasoning: llmResponse,
      });

      // ── Execute action in game engine ─────────────────────────────
      try {
        frame = await bridge.sendAction(action);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        sseManager.sendEvent(config.evalId, 'log', {
          level: 'error',
          message: `[${game.gameId}] [${modelId}] Action error: ${msg}`,
        });
        break;
      }

      // Send result with updated game state
      sseManager.sendEvent(config.evalId, 'turn.result', {
        gameId: game.gameId,
        modelId,
        runIndex,
        turn,
        action,
        frame: {
          grid: frame.grid,
          width: frame.width,
          height: frame.height,
          state: frame.state,
          level: frame.level,
          available_actions: frame.available_actions,
        },
        score: frame.level || 0,
      });

      // Small delay for UI readability and to avoid API throttling
      await new Promise((r) => setTimeout(r, 300));
    }

    // ── Record run result ─────────────────────────────────────────────
    const totalTime = (Date.now() - startTime) / 1000;
    const result: EvalRunResult = {
      gameId: game.gameId,
      modelId,
      runIndex,
      score: frame.level || 0,
      state: frame.state || 'NOT_FINISHED',
      totalTurns: actions.length,
      totalTime,
      actions,
    };
    session.results.push(result);

    sseManager.sendEvent(config.evalId, 'run.completed', result);
    sseManager.sendEvent(config.evalId, 'log', {
      level: 'info',
      message: `[${game.gameId}] [${modelId}] Run ${runIndex + 1} complete: ${frame.state} in ${actions.length} turns (${totalTime.toFixed(1)}s)`,
    });
  } finally {
    bridge.kill();
  }
}

// ── Helper: Build system prompt ───────────────────────────────────────────

/**
 * Construct the system prompt from the user's base prompt and reasoning effort.
 * Includes colour code legend, strategy guidance, notepad instructions, and
 * the strict response format.
 */
function buildSystemPrompt(
  basePrompt: string,
  effort: 'low' | 'medium' | 'high',
): string {
  const colorLegend = [
    'Grid colour codes:',
    '   0 = black (empty)    1 = blue       2 = red        3 = green',
    '   4 = yellow           5 = grey       6 = pink       7 = orange',
    '   8 = teal             9 = brown     10 = dark blue  11 = dark red',
    '  12 = dark green      13 = dark yellow 14 = dark grey 15 = dark pink',
  ].join('\n');

  const strategyByEffort: Record<typeof effort, string> = {
    low: [
      'Strategy (concise):',
      '  - Act quickly. Pick the action most likely to increase your score.',
      '  - Keep reasoning to 1-2 sentences.',
    ].join('\n'),
    medium: [
      'Strategy (balanced):',
      '  - Study the grid patterns briefly before acting.',
      '  - Look for colour clusters, symmetry, and repeating motifs.',
      '  - Prefer actions that increase the score or reveal information.',
      '  - Keep reasoning to 2-3 sentences.',
    ].join('\n'),
    high: [
      'Strategy (thorough):',
      '  - Carefully analyse the grid: identify colour patterns, symmetry, object boundaries, and spatial relationships.',
      '  - Form and test hypotheses about the puzzle mechanics.',
      '  - Track which actions change the grid and how, using your notepad.',
      '  - Consider all available actions before choosing.',
      '  - Reason in 3-5 sentences before selecting your action.',
    ].join('\n'),
  };

  const notepadInstructions = [
    'Notepad system:',
    '  You have a persistent notepad to record observations, hypotheses, and plans.',
    '  Update it by including a <NOTEPAD>your notes here</NOTEPAD> block anywhere in your response.',
    '  Your notepad content is shown back to you each turn. Use it to track:',
    '    - Patterns you have identified',
    '    - Which actions do what',
    '    - Your current strategy and what to try next',
    '  The notepad resets at the start of each new run.',
  ].join('\n');

  const responseFormat = [
    'Response format (STRICT):',
    '  1. Reasoning: explain what you see and why you chose your action.',
    '  2. Optionally update your notepad: <NOTEPAD>updated notes</NOTEPAD>',
    '  3. Last line must be exactly: ACTION: <action_name>',
    '     where <action_name> is one of the available action names (e.g. ACTION2, RESET).',
    '  Do NOT add anything after the ACTION line.',
  ].join('\n');

  const parts: string[] = [];

  // Include user's base prompt first if provided
  if (basePrompt && basePrompt.trim()) {
    parts.push(basePrompt.trim());
    parts.push('');
  } else {
    parts.push('You are an AI agent playing a visual grid-based puzzle game (ARC-AGI format).');
    parts.push('The goal is to complete all levels before running out of actions.');
    parts.push('');
  }

  parts.push(colorLegend);
  parts.push('');
  parts.push(strategyByEffort[effort]);
  parts.push('');
  parts.push(notepadInstructions);
  parts.push('');
  parts.push(responseFormat);

  return parts.join('\n');
}

// ── Helper: Build turn message ────────────────────────────────────────────

/**
 * Format the user turn message with current game state, grid, available
 * actions, notepad, and instructions.
 */
function buildTurnMessage(
  frame: GameFrame,
  turn: number,
  notepad: string,
  maxSteps: number,
): string {
  // Determine win target from frame metadata if available
  const winLevels = (frame as any).metadata?.level_count
    ?? (frame as any).win_score
    ?? '?';

  // Format grid rows
  const gridLines = frame.grid.map(
    (row, i) => `Row ${String(i).padStart(2)}: [${row.join(', ')}]`,
  );

  const availableStr = (frame.available_actions ?? []).join(', ')
    || 'ACTION1, ACTION2, ACTION3, ACTION4, ACTION5';

  const notepadSection = notepad
    ? `Your notepad:\n${notepad}`
    : 'Your notepad: (empty)';

  return [
    `=== Turn ${turn + 1} ===`,
    `Levels completed: ${frame.level} / ${winLevels}`,
    `Actions used: ${turn + 1}/${maxSteps}`,
    `Game status: ${frame.state}`,
    '',
    'Current grid:',
    ...gridLines,
    '',
    `Available actions: ${availableStr}`,
    '',
    notepadSection,
    '',
    'Reason briefly, optionally update <NOTEPAD>...</NOTEPAD>, then:',
    'ACTION: <action_name>',
  ].join('\n');
}

// ── Helper: Extract action from LLM response ─────────────────────────────

/**
 * Parse the chosen action from the model's text response.
 *
 * Priority:
 *   1. Explicit labeled pattern: "ACTION: ACTION2" or "ACTION: RESET"
 *   2. Last bare action token in the text (ACTION1-7 or RESET)
 *   3. Fallback to the first available action
 */
function extractAction(text: string, availableActions: string[]): string {
  const upper = text.toUpperCase();

  // Priority 1: Labeled "ACTION: <token>"
  const labeled = upper.match(/ACTION\s*:\s*(ACTION[1-7]|RESET)/);
  if (labeled) return labeled[1];

  // Priority 2: Last bare action token
  const bare = [...upper.matchAll(/\b(ACTION[1-7]|RESET)\b/g)];
  if (bare.length > 0) return bare[bare.length - 1][1];

  // Priority 3: Fallback to first available action
  const first = availableActions.find(Boolean) ?? 'ACTION1';
  return first.toUpperCase();
}

// ── Helper: Extract notepad ───────────────────────────────────────────────

/**
 * Extract updated notepad content from the model's response.
 * Looks for <NOTEPAD>...</NOTEPAD> XML block (case-insensitive).
 * Returns null if no notepad block is found (caller preserves previous notepad).
 */
function extractNotepad(text: string): string | null {
  const match = text.match(/<NOTEPAD>([\s\S]*?)<\/NOTEPAD>/i);
  if (match) return match[1].trim();
  return null;
}

// ── Helper: Trim conversation history ─────────────────────────────────────

/**
 * Keep the system message plus the last `windowSize` user/assistant turn pairs.
 * Mutates the messages array in place.
 *
 * Layout: [system, user, assistant, user, assistant, ...]
 * We always keep messages[0] (system). From the rest we keep the last
 * windowSize * 2 entries (each turn pair = 1 user + 1 assistant message).
 */
function trimHistory(messages: LLMMessage[], windowSize: number): void {
  // messages[0] is always the system prompt
  // The rest are user/assistant pairs
  const conversationMessages = messages.length - 1; // exclude system
  const maxConversation = windowSize * 2;

  if (conversationMessages > maxConversation) {
    // Remove the oldest pairs, keeping system[0] + last maxConversation messages
    const removeCount = conversationMessages - maxConversation;
    messages.splice(1, removeCount);
  }
}
