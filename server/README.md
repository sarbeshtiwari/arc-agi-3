# Server — ARC-AGI-3 Internal Platform

## Overview

Express.js + TypeScript server that serves both the React frontend and the REST API. Uses PostgreSQL for persistence and spawns long-lived Python subprocesses to run ARC game engines via an NDJSON bridge protocol. Authentication is JWT-based with bcrypt password hashing and role-based access control (RBAC).

## File Structure

```
server/
├── index.ts                    # Express bootstrap, middleware, Vite integration
├── vite.ts                     # Vite dev server middleware (SPA fallback)
├── routes.ts                   # Router registry + startup init (initDB, ensureDefaultAdmin)
├── db.ts                       # PostgreSQL pool (pg), schema DDL, query helpers
├── middleware/
│   ├── auth.ts                 # JWT auth, bcrypt, RBAC (authenticateToken, requireAdmin, requirePage)
│   ├── asyncHandler.ts         # Async error wrapper for route handlers
│   └── errorHandler.ts         # AppError class + Express error handler
├── routes/
│   ├── auth.ts                 # 3 endpoints  — login, me, register
│   ├── games.ts                # 20 endpoints — public listing/stats/preview/video, admin CRUD/upload/sync
│   ├── player.ts               # 12 endpoints — public/ephemeral/admin game sessions + palette
│   ├── analytics.ts            # 9 endpoints  — dashboard, per-game analytics, sessions, exports (Excel)
│   ├── users.ts                # 5 endpoints  — list, get, update, delete, protected password change
│   └── requests.ts             # 7 endpoints  — public submit, admin list/get/source/files/review/delete
├── services/
│   └── GamePythonBridge.ts     # NDJSON subprocess bridge to Python game_runner.py
└── python/
    └── game_runner.py          # Python process: loads ARCEngine games, executes actions
```

## Startup Sequence

1. **Load environment** — `dotenv/config` reads `.env`
2. **Express middleware** — `cors` (credentials), `json` (50 MB limit), `urlencoded`, `cookieParser`, trust proxy, request logger (logs `/api` routes with method/path/status/ms)
3. **`registerRoutes(app)`** — runs sequentially:
   - `initDB()` — connects to PostgreSQL, runs `CREATE TABLE IF NOT EXISTS` for all 6 tables (`users`, `games`, `play_sessions`, `game_analytics`, `game_requests`, `temp_game_sessions`) plus indexes
   - `ensureDefaultAdmin()` — creates the default admin user (from `DEFAULT_ADMIN_USERNAME`/`DEFAULT_ADMIN_PASSWORD` env vars) and an optional protected super admin (from `PROTECTED_USERNAME`/`PROTECTED_SECRET_CODE`)
   - Mounts routers: `/api/auth`, `/api/games`, `/api/player`, `/api/analytics`, `/api/users`, `/api/requests`
   - Mounts the `errorHandler` middleware (must be last)
   - Returns an `http.Server`
4. **Dev mode** — `setupVite()` creates a Vite dev server in middleware mode, attaches HMR to the HTTP server, and serves the SPA with cache-busted `main.tsx`
5. **Prod mode** — serves `dist/public/` as static, with a catch-all `*` route that returns `index.html` for non-`/api` paths
6. **Listen** — binds to `0.0.0.0:${PORT}` (default 5000)

## Database Schema

6 tables, all with `TEXT` primary keys (UUIDs via `crypto.randomUUID()`):

| Table | Purpose |
|---|---|
| `users` | Admin/player accounts with RBAC (`is_admin`, `allowed_pages` JSONB) |
| `games` | Game registry — metadata, file paths, play/win counters |
| `play_sessions` | Per-play records — state, actions, time, level stats (JSONB) |
| `game_analytics` | Aggregated daily analytics per game |
| `game_requests` | Community game submissions (pending/approved/rejected) with BYTEA file blobs |
| `temp_game_sessions` | Ephemeral sessions from direct-play (auto-expire after 1 day) |

## Session Tracking (In-Memory)

The `SessionTracker` interface (defined in `server/routes/player.ts`) tracks real-time game state per active session in a `Map<string, SessionTracker>`. This data lives in memory alongside the Python bridge and is persisted to the database after each action.

**What it tracks:**

| Field | Purpose |
|---|---|
| `startTime` | Session start timestamp (epoch ms) |
| `levelStartTime` | When the current level began |
| `levelStartActions` | Action count at level start (for computing per-level actions) |
| `currentLevel` | Current level index (0-based) |
| `completedLevels` | Array of `{ level, actions, time, completed, game_overs, resets }` |
| `actionsSinceReset` | Counter reset to 0 on every RESET action |
| `levelGameOvers` | GAME_OVER events on the current level |
| `levelResets` | Manual resets (R key) on the current level |
| `totalGameOvers` | Session-wide GAME_OVER count |
| `totalResets` | Session-wide reset count |
| `prevState` | Previous frame state (for transition detection) |

**Level transition detection:**

After each action, the tracker compares `frame.level` against `tracker.currentLevel`. When `frame.level > currentLevel`:
1. The completed level's stats are computed (actions = `frame.total_actions - levelStartActions`, time = elapsed since `levelStartTime`)
2. Stats are pushed to `completedLevels` with per-level `game_overs` and `resets`
3. Per-level counters are reset for the new level

**GAME_OVER event counting:**

A GAME_OVER is counted only on _transition_ — when `frame.state === "GAME_OVER"` and `prevState !== "GAME_OVER"`. This prevents double-counting from repeated frames in the same GAME_OVER state.

**Double-reset logic:**

When the player sends a RESET action:
1. `levelResets++`, `totalResets++`
2. If `actionsSinceReset === 0` (i.e., two consecutive resets with no actions in between), the session is fully restarted: `completedLevels` is cleared, `currentLevel` set to 0, `startTime` reset, per-level counters zeroed
3. `actionsSinceReset` is set to 0, `levelStartTime` and `levelStartActions` are reset

This means pressing R once resets the current level, pressing R twice with no actions in between restarts from level 1.

## Python Bridge Protocol

`GamePythonBridge` (in `server/services/GamePythonBridge.ts`) spawns `game_runner.py` as a long-lived subprocess communicating via NDJSON (newline-delimited JSON) over stdin/stdout. The protocol is strictly request-response: one command produces exactly one JSON response.

### Commands (Node → Python via stdin)

**init** — Load a game and get the initial frame:
```json
→ {"command":"init","game_id":"fm01","game_path":"environment_files/fm01/v1/fm01.py","seed":0}
← {"type":"ready","game_id":"fm01","frame":{"type":"frame","grid":[[...]],"width":64,"height":64,"state":"NOT_FINISHED","level":0,"levels_completed":0,"total_actions":0,"max_actions":100,"available_actions":["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5"],"win_levels":0},"metadata":{"game_id":"fm01","level_count":5,"win_score":1,"max_actions":100}}
```

**action** — Perform a game action:
```json
→ {"command":"action","action":"ACTION5"}
← {"type":"frame","grid":[[...]],"width":64,"height":64,"state":"NOT_FINISHED","level":0,"levels_completed":0,"total_actions":1,"max_actions":100,"available_actions":["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5"],"win_levels":0}
```

**action with coordinates** (ACTION6 only):
```json
→ {"command":"action","action":"ACTION6","x":10,"y":5}
← {"type":"frame",...}
```

**reset** — Reset game to initial state (resets `total_actions` to 0):
```json
→ {"command":"reset"}
← {"type":"frame","grid":[[...]],"state":"NOT_FINISHED","level":0,"total_actions":0,...}
```

**quit** — Graceful shutdown:
```json
→ {"command":"quit"}
← {"type":"quit","message":"goodbye"}
```

### Error Responses

```json
← {"type":"error","code":"NO_GAME","message":"No game loaded. Send 'init' first."}
← {"type":"error","code":"FILE_NOT_FOUND","message":"Game file not found: ..."}
← {"type":"error","code":"INVALID_GAME","message":"No ARCBaseGame subclass found in ..."}
```

Error codes: `ARCENGINE_NOT_FOUND`, `INVALID_JSON`, `MISSING_GAME_PATH`, `NO_GAME`, `FILE_NOT_FOUND`, `IMPORT_ERROR`, `INVALID_GAME`, `UNKNOWN_COMMAND`, `GAME_ERROR`, `UNEXPECTED_ERROR`.

### Bridge Lifecycle

- **Spawn**: lazy, on first `init()` call. Uses `PYTHON_BIN` env var, `.python-bin` marker file, or system `python3`.
- **Timeout**: 30 seconds per command (configurable via `DEFAULT_TIMEOUT_MS`)
- **Kill**: sends `{"command":"quit"}` via stdin, then `SIGTERM`. Cleans up readline interface.
- **Stderr**: captured and logged as `[GamePythonBridge stderr]`

## Database Query Patterns

### JSONB writes — always use `toJsonb()`
```ts
import { toJsonb } from "../db.js";

// Writing arrays/objects to JSONB columns:
await pool.query(
  `UPDATE play_sessions SET level_stats = $1 WHERE session_guid = $2`,
  [toJsonb(completedLevels), sessionGuid]
);

// toJsonb handles null, undefined, already-stringified, and objects:
toJsonb(null)            // → null
toJsonb([1, 2, 3])       // → "[1,2,3]"
toJsonb("already json")  // → "already json" (passthrough)
```

### JSONB reads — native JS objects
PostgreSQL's `pg` driver automatically parses JSONB columns into JavaScript objects/arrays. No manual `JSON.parse()` needed:
```ts
const row = await queryOne("SELECT level_stats FROM play_sessions WHERE ...");
// row.level_stats is already a JS array, not a string
```

### Booleans
PostgreSQL returns real `true`/`false` for `BOOLEAN` columns (not `"t"`/`"f"`):
```ts
if (user.is_admin) { ... }  // works directly
```

### Parameterized queries
Always use `$1, $2, $3` positional syntax:
```ts
await pool.query(
  "SELECT * FROM games WHERE game_id = $1 AND is_active = $2",
  [gameId, true]
);
```

### NULL safety with COALESCE
For atomic increments on potentially-NULL columns:
```ts
await pool.query(
  `UPDATE games SET total_plays = COALESCE(total_plays, 0) + 1 WHERE id = $1`,
  [gameDbId]
);
```

### Query helpers
```ts
import pool, { genId, queryOne, queryAll, toJsonb } from "../db.js";

genId()                          // crypto.randomUUID()
queryOne(sql, params)            // returns first row or null
queryAll(sql, params)            // returns all rows as array
pool.query(sql, params)          // raw pg query (returns { rows, rowCount })
```

## Error Handling

Three layers work together to catch and format all errors:

### 1. `AppError` (known/expected errors)
Thrown in route handlers for business logic errors:
```ts
import { AppError } from "../middleware/errorHandler.js";

throw new AppError("Game not found", 404);
throw new AppError("metadata.json must contain 'game_id' field", 400);
throw new AppError("Admin access required", 403);
```

### 2. `asyncHandler` (async route wrapper)
Wraps every async route handler so rejected promises are forwarded to `next()`:
```ts
import { asyncHandler } from "../middleware/asyncHandler.js";

router.get("/", asyncHandler(async (req, res) => {
  // If this throws, the error handler catches it
  const data = await queryAll("SELECT ...");
  res.json(data);
}));
```

### 3. `errorHandler` (Express error middleware)
Mounted as the last middleware. Returns consistent JSON responses:
```ts
// AppError → uses statusCode + message
// Other errors → uses err.statusCode || err.status || 500

// Response format:
{ "detail": "Error message here" }
```

All errors are logged to console as `[ERROR] METHOD /path: message`.

## API Endpoints Summary

| Router | Mount Point | Public | Auth | Admin | Total |
|--------|-------------|--------|------|-------|-------|
| auth | `/api/auth` | 1 (login) | 1 (me) | 1 (register) | **3** |
| games | `/api/games` | 7 | 0 | 13 | **20** |
| player | `/api/player` | 4 (3 public + palette) | 4 | 4 | **12** |
| analytics | `/api/analytics` | 0 | 4 | 5 | **9** |
| users | `/api/users` | 0 | 1 (protected pw) | 4 | **5** |
| requests | `/api/requests` | 1 (submit) | 0 | 6 | **7** |
| — | `/api/health` | 1 | 0 | 0 | **1** |
| | | **14** | **10** | **33** | **57** |
