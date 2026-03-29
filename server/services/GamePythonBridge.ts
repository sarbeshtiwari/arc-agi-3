/**
 * GamePythonBridge — spawns and manages the Python game_runner.py subprocess.
 *
 * Communicates with the Python process via NDJSON (newline-delimited JSON)
 * over stdin/stdout. Each command sent to stdin produces exactly one JSON
 * response on stdout.
 *
 * Commands: init, action, reset, quit
 * Responses: ready, frame, error, quit
 */

import { spawn, type ChildProcess } from "child_process";
import * as path from "path";
import * as readline from "readline";
import * as fs from "fs";
import logService from "./LogService.js";

// Resolve Python runner path relative to cwd (works in both dev and prod)
const PYTHON_RUNNER_PATH = path.join(
  process.cwd(),
  "server",
  "python",
  "game_runner.py",
);

const DEFAULT_TIMEOUT_MS = 30_000; // 30 seconds

/** Platform-aware Python binary. Priority: PYTHON_BIN env > .python-bin marker > system python3 */
function resolvePythonBin(): string {
  if (process.env.PYTHON_BIN) {
    logService.debug("bridge", "Python binary from PYTHON_BIN env", { path: process.env.PYTHON_BIN });
    return process.env.PYTHON_BIN;
  }
  const marker = path.join(process.cwd(), ".python-bin");
  if (fs.existsSync(marker)) {
    const bin = fs.readFileSync(marker, "utf-8").trim();
    if (bin) {
      logService.debug("bridge", "Python binary from .python-bin marker", { path: bin });
      return bin;
    }
  }
  const fallback = process.platform === "win32" ? "python" : "python3";
  logService.debug("bridge", "Python binary using system fallback", { path: fallback });
  return fallback;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GameFrame {
  grid: number[][];
  width: number;
  height: number;
  state: string;
  level: number;
  total_actions: number;
  available_actions: string[];
  reward: number;
  done: boolean;
  text_observation: string;
  image_observation_b64: string | null;
  metadata: Record<string, any>;
  [key: string]: any;
}

interface ReadyResponse {
  type: "ready";
  game_id: string;
  frame: GameFrame;
  metadata: {
    game_id: string;
    level_count: number;
    total_levels: number;
    action_map_inferred?: boolean;
  };
}

interface FrameResponse {
  type: "frame";
  [key: string]: any;
}

interface ErrorResponse {
  type: "error";
  code: string;
  message: string;
}

interface QuitResponse {
  type: "quit";
  message: string;
}

type BridgeMessage = ReadyResponse | FrameResponse | ErrorResponse | QuitResponse;

// ---------------------------------------------------------------------------
// GamePythonBridge
// ---------------------------------------------------------------------------

export class GamePythonBridge {
  private proc: ChildProcess | null = null;
  private rl: readline.Interface | null = null;
  private killed = false;
  private stderrChunks: string[] = [];

  /**
   * The pending promise resolver for the current in-flight request.
   * The protocol is strictly request-response (one command → one reply),
   * so at most one resolver is active at any time.
   */
  private pending: {
    resolve: (msg: BridgeMessage) => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | null = null;

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  /** Spawn the Python subprocess. Called lazily on first init(). */
  private ensureProcess(): void {
    if (this.proc) return;

    const pythonBin = resolvePythonBin();
    logService.debug("bridge", "Spawning Python subprocess", {
      python_bin: pythonBin,
      runner_path: PYTHON_RUNNER_PATH,
      cwd: process.cwd(),
    });

    this.proc = spawn(pythonBin, [PYTHON_RUNNER_PATH], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    logService.info("bridge", "Python subprocess spawned", { pid: this.proc.pid });

    this.proc.stderr?.on("data", (chunk: Buffer) => {
      const text = chunk.toString().trim();
      if (text) {
        this.stderrChunks.push(text);
        logService.warn("bridge", `Python stderr: ${text}`, { pid: this.proc?.pid });
      }
    });

    this.rl = readline.createInterface({
      input: this.proc.stdout!,
      terminal: false,
    });

    this.rl.on("line", (line: string) => {
      this.handleLine(line);
    });

    this.proc.on("exit", (code, signal) => {
      if (!this.killed) {
        logService.error("bridge", "Python process exited unexpectedly", {
          code, signal, pid: this.proc?.pid,
          stderr_tail: this.stderrChunks.slice(-5).join("\n"),
        });
      } else {
        logService.debug("bridge", "Python process exited (killed)", { code, signal });
      }
      this.rejectPending(
        new Error(
          `Python process exited (code=${code}, signal=${signal})`,
        ),
      );
      this.rl?.close();
      this.rl = null;
      this.proc = null;
    });

    this.proc.on("error", (err) => {
      logService.error("bridge", `Failed to spawn Python: ${err.message}`, {
        python_bin: pythonBin,
        error: err.message,
      });
      this.rejectPending(err);
      this.proc = null;
    });
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Initialise a game. Loads the game file in Python and returns the initial frame.
   */
  async init(
    gameId: string,
    gamePath: string,
    seed?: number,
  ): Promise<GameFrame> {
    logService.debug("bridge", "init() called", { game_id: gameId, game_path: gamePath, seed });
    this.ensureProcess();

    const msg = await this.send({
      command: "init",
      game_id: gameId,
      game_path: gamePath,
      ...(seed !== undefined && { seed }),
    });

    if (msg.type === "ready") {
      const ready = msg as ReadyResponse;
      const frame = ready.frame;
      if (ready.metadata?.action_map_inferred) {
        frame.action_map_inferred = true;
      }
      logService.info("bridge", "Game initialized via bridge", {
        game_id: gameId,
        grid_size: `${frame.height}x${frame.width}`,
        available_actions: frame.available_actions,
        level: frame.level,
        action_map_inferred: !!ready.metadata?.action_map_inferred,
        pid: this.proc?.pid,
      });
      return frame;
    }

    if (msg.type === "error") {
      const err = msg as ErrorResponse;
      logService.error("bridge", `Init failed: ${err.message}`, { game_id: gameId, code: err.code });
      throw new Error(`Init failed [${err.code}]: ${err.message}`);
    }

    throw new Error(`Unexpected response type during init: ${msg.type}`);
  }

  /**
   * Send a game action and return the resulting frame.
   */
  async sendAction(
    action: string,
    x?: number,
    y?: number,
  ): Promise<GameFrame> {
    logService.debug("bridge", `Sending action: ${action}`, { action, x, y });
    const payload: Record<string, any> = { command: "action", action };
    if (x !== undefined && y !== undefined) {
      payload.x = x;
      payload.y = y;
    }

    const msg = await this.send(payload);

    if (msg.type === "frame") {
      const frame = msg as unknown as GameFrame;
      logService.debug("bridge", `Action response received`, {
        action,
        state: frame.state,
        level: frame.level,
        reward: frame.reward,
        done: frame.done,
        total_actions: frame.total_actions,
        grid_size: `${frame.height}x${frame.width}`,
      });
      return frame;
    }

    if (msg.type === "error") {
      const err = msg as ErrorResponse;
      logService.error("bridge", `Action failed: ${err.message}`, { action, code: err.code });
      throw new Error(`Action failed [${err.code}]: ${err.message}`);
    }

    throw new Error(`Unexpected response type during action: ${msg.type}`);
  }

  /**
   * Reset the game to its initial state and return the frame.
   */
  async reset(): Promise<GameFrame> {
    logService.debug("bridge", "Sending reset command");
    const msg = await this.send({ command: "reset" });

    if (msg.type === "frame") {
      const frame = msg as unknown as GameFrame;
      logService.debug("bridge", "Reset response received", {
        state: frame.state,
        level: frame.level,
        grid_size: `${frame.height}x${frame.width}`,
        available_actions: frame.available_actions,
      });
      return frame;
    }

    if (msg.type === "error") {
      const err = msg as ErrorResponse;
      logService.error("bridge", `Reset failed: ${err.message}`, { code: err.code });
      throw new Error(`Reset failed [${err.code}]: ${err.message}`);
    }

    throw new Error(`Unexpected response type during reset: ${msg.type}`);
  }

  /**
   * Kill the Python subprocess and clean up all resources.
   */
  kill(): void {
    logService.debug("bridge", "Killing bridge", { pid: this.proc?.pid });
    this.killed = true;
    this.rejectPending(new Error("Bridge killed"));

    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }

    if (this.proc) {
      // Try graceful quit first, then SIGTERM
      try {
        this.proc.stdin?.write(JSON.stringify({ command: "quit" }) + "\n");
      } catch {
        // stdin may already be closed
      }
      try {
        this.proc.kill("SIGTERM");
      } catch {
        // process may already be dead
      }
      this.proc = null;
    }
  }

  // -----------------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------------

  /**
   * Send a command to the Python process and wait for the next JSON response.
   */
  private send(payload: Record<string, any>): Promise<BridgeMessage> {
    return new Promise((resolve, reject) => {
      if (!this.proc || !this.proc.stdin) {
        reject(new Error("Python process not running"));
        return;
      }

      if (this.killed) {
        reject(new Error("Bridge has been killed"));
        return;
      }

      if (this.pending) {
        reject(new Error("Another request is already in flight"));
        return;
      }

      const timer = setTimeout(() => {
        this.pending = null;
        logService.error("bridge", `Command timed out after ${DEFAULT_TIMEOUT_MS}ms`, { command: payload.command });
        reject(new Error(`Command timed out after ${DEFAULT_TIMEOUT_MS}ms: ${payload.command}`));
      }, DEFAULT_TIMEOUT_MS);

      this.pending = { resolve, reject, timer };

      const line = JSON.stringify(payload) + "\n";
      logService.debug("bridge", `Sending command to Python`, { command: payload.command, payload_size: line.length });
      this.proc.stdin.write(line, (err) => {
        if (err) {
          clearTimeout(timer);
          this.pending = null;
          logService.error("bridge", `Failed to write to stdin: ${err.message}`);
          reject(new Error(`Failed to write to stdin: ${err.message}`));
        }
      });
    });
  }

  /**
   * Handle a single line of NDJSON from the Python process.
   */
  private handleLine(line: string): void {
    if (!line.trim()) return;

    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      logService.warn("bridge", `Failed to parse NDJSON line`, { line: line.substring(0, 200) });
      return;
    }

    if (msg.type === "log") {
      const level = msg.level || "debug";
      const logMsg = msg.message || "";
      const meta = { ...msg.metadata, source_python: true, pid: this.proc?.pid };
      logService.log(level, "python", logMsg, meta);
      return;
    }

    if (this.pending) {
      const { resolve, timer } = this.pending;
      clearTimeout(timer);
      this.pending = null;
      resolve(msg);
    } else {
      logService.warn("bridge", "Received message with no pending request", { type: msg.type });
    }
  }

  /**
   * Reject the current pending promise if one exists.
   */
  private rejectPending(err: Error): void {
    if (this.pending) {
      clearTimeout(this.pending.timer);
      const { reject } = this.pending;
      this.pending = null;
      reject(err);
    }
  }
}
