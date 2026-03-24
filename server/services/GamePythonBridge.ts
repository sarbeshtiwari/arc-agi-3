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
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  // Check marker file written by postinstall script
  const marker = path.join(process.cwd(), ".python-bin");
  if (fs.existsSync(marker)) {
    const bin = fs.readFileSync(marker, "utf-8").trim();
    if (bin) return bin;
  }
  return process.platform === "win32" ? "python" : "python3";
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
  [key: string]: any;
}

interface ReadyResponse {
  type: "ready";
  game_id: string;
  frame: GameFrame;
  metadata: {
    game_id: string;
    level_count: number;
    win_score: number;
    max_actions: number;
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

    this.proc = spawn(resolvePythonBin(), [PYTHON_RUNNER_PATH], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    // Capture stderr for diagnostics
    this.proc.stderr?.on("data", (chunk: Buffer) => {
      const text = chunk.toString().trim();
      if (text) {
        this.stderrChunks.push(text);
        console.error(`[GamePythonBridge stderr] ${text}`);
      }
    });

    // Parse NDJSON from stdout
    this.rl = readline.createInterface({
      input: this.proc.stdout!,
      terminal: false,
    });

    this.rl.on("line", (line: string) => {
      this.handleLine(line);
    });

    // If the process exits unexpectedly, reject any pending promise
    this.proc.on("exit", (code, signal) => {
      if (!this.killed) {
        console.error(
          `[GamePythonBridge] Python process exited unexpectedly (code=${code}, signal=${signal})`,
        );
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
      console.error(`[GamePythonBridge] Failed to spawn Python: ${err.message}`);
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
    this.ensureProcess();

    const msg = await this.send({
      command: "init",
      game_id: gameId,
      game_path: gamePath,
      ...(seed !== undefined && { seed }),
    });

    if (msg.type === "ready") {
      return (msg as ReadyResponse).frame;
    }

    if (msg.type === "error") {
      throw new Error(`Init failed [${(msg as ErrorResponse).code}]: ${(msg as ErrorResponse).message}`);
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
    const payload: Record<string, any> = { command: "action", action };
    if (x !== undefined && y !== undefined) {
      payload.x = x;
      payload.y = y;
    }

    const msg = await this.send(payload);

    if (msg.type === "frame") {
      return msg as unknown as GameFrame;
    }

    if (msg.type === "error") {
      throw new Error(`Action failed [${(msg as ErrorResponse).code}]: ${(msg as ErrorResponse).message}`);
    }

    throw new Error(`Unexpected response type during action: ${msg.type}`);
  }

  /**
   * Reset the game to its initial state and return the frame.
   */
  async reset(): Promise<GameFrame> {
    const msg = await this.send({ command: "reset" });

    if (msg.type === "frame") {
      return msg as unknown as GameFrame;
    }

    if (msg.type === "error") {
      throw new Error(`Reset failed [${(msg as ErrorResponse).code}]: ${(msg as ErrorResponse).message}`);
    }

    throw new Error(`Unexpected response type during reset: ${msg.type}`);
  }

  /**
   * Kill the Python subprocess and clean up all resources.
   */
  kill(): void {
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

      // Reject if there's already a pending request (shouldn't happen with correct usage)
      if (this.pending) {
        reject(new Error("Another request is already in flight"));
        return;
      }

      const timer = setTimeout(() => {
        this.pending = null;
        reject(new Error(`Command timed out after ${DEFAULT_TIMEOUT_MS}ms: ${payload.command}`));
      }, DEFAULT_TIMEOUT_MS);

      this.pending = { resolve, reject, timer };

      const line = JSON.stringify(payload) + "\n";
      this.proc.stdin.write(line, (err) => {
        if (err) {
          clearTimeout(timer);
          this.pending = null;
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

    let msg: BridgeMessage;
    try {
      msg = JSON.parse(line);
    } catch {
      console.error(`[GamePythonBridge] Failed to parse line: ${line}`);
      return;
    }

    if (this.pending) {
      const { resolve, timer } = this.pending;
      clearTimeout(timer);
      this.pending = null;
      resolve(msg);
    } else {
      // No pending request — log unexpected message
      console.warn(`[GamePythonBridge] Received message with no pending request: ${line}`);
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
