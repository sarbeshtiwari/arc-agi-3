# ARC-AGI-3 Internal Platform

Internal full-stack platform for managing, playtesting, and analyzing ARC-AGI-3 grid-based puzzle games. Provides a public homepage where players can browse and play active games, an ephemeral "Play Your Own" mode for testing uploaded game files, a game request submission workflow with admin approval, an admin panel with analytics dashboards and Excel exports, per-level session tracking, and optional screen recording with server upload. Built as a single-port Node.js/Express + React + Vite monorepo, with Python running only as a subprocess for the game engine.

---

## Architecture

```
Single-port monorepo: Express (API + static) + Vite (React HMR) + Python subprocess

┌──────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  React SPA (Vite + React Router + Tailwind)                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                   HTTP :5000 (single port)
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Express Server                             │
│                                                              │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  /api/*        │  │  Vite Dev     │  │  Static Files  │  │
│  │  REST routes   │  │  Middleware   │  │  (prod only)   │  │
│  │  (6 routers)   │  │  + HMR (dev) │  │  dist/public/  │  │
│  └───────┬────────┘  └───────────────┘  └────────────────┘  │
│          │                                                   │
│  ┌───────▼────────────────────────────────────────────────┐  │
│  │  GamePythonBridge (per-session child process)          │  │
│  │  spawn(python3, [game_runner.py])                      │  │
│  │  stdin → NDJSON commands                               │  │
│  │  stdout ← NDJSON responses                             │  │
│  └───────┬────────────────────────────────────────────────┘  │
│          │                                                   │
│  ┌───────▼──────┐  ┌─────────────────────────────────────┐  │
│  │  PostgreSQL   │  │  environment_files/                 │  │
│  │  6 tables     │  │  <game_code>/<version>/game.py      │  │
│  │              │  │  <game_code>/<version>/metadata.json │  │
│  └──────────────┘  └─────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Dev mode vs Prod mode

| Concern | Development (`npm run dev`) | Production (`npm start`) |
|---------|---------------------------|--------------------------|
| Frontend | Vite dev server as Express middleware; React HMR via WebSocket on same port | Pre-built static files served from `dist/public/` |
| Server | `tsx` loader runs TypeScript directly | `esbuild` bundles server to `dist/index.js` |
| SPA fallback | Vite transforms `client/index.html` on the fly | Express serves `dist/public/index.html` for non-API routes |

### Python is NOT a server

Python runs only as a child process (`child_process.spawn`) managed by `GamePythonBridge`. Each active play session gets its own Python subprocess running `server/python/game_runner.py`. The bridge communicates over stdin/stdout using newline-delimited JSON (NDJSON). When a session ends or the bridge is killed, the Python process exits. There is no Python HTTP server, no Flask, no FastAPI.

---

## Quick Start

### Prerequisites

- **Node.js 20+** (ESM support required; `type: "module"` in package.json)
- **PostgreSQL 14+** running locally or remotely
- **Python 3.10+** with the `arc-agi` package (provides `arcengine`)

### Setup

```bash
git clone <repo-url> arc-agi-internal
cd arc-agi-internal
npm install        # also runs postinstall → auto-detects Python, installs arc-agi
npm run dev        # starts Express + Vite on http://localhost:5000
```

On first start:
1. All 6 database tables are created automatically via `initDB()`.
2. A default admin user is created from `DEFAULT_ADMIN_USERNAME` / `DEFAULT_ADMIN_PASSWORD` env vars (defaults: `admin` / `admin123`).
3. If `PROTECTED_USERNAME` is set, a protected super admin is also created.

### Environment Variables

Create a `.env` file in the project root:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `5000` | Server listen port |
| `NODE_ENV` | No | `development` | `development` or `production` |
| `DB_HOST` | No | `localhost` | PostgreSQL host |
| `DB_PORT` | No | `5432` | PostgreSQL port |
| `DB_NAME` | No | `arc_agi_db` | PostgreSQL database name |
| `DB_USER` | No | `arcadmin` | PostgreSQL username |
| `DB_PASSWORD` | No | `arcadmin` | PostgreSQL password |
| `SECRET_KEY` | **Yes** | `change-me` | JWT signing secret (change in production) |
| `JWT_EXPIRES_IN` | No | `1440m` | JWT token expiry (24 hours) |
| `DEFAULT_ADMIN_USERNAME` | No | `admin` | Default admin account username |
| `DEFAULT_ADMIN_PASSWORD` | No | `admin123` | Default admin account password |
| `PROTECTED_USERNAME` | No | *(none)* | Username for protected super admin account |
| `PROTECTED_SECRET_CODE` | No | `superadmin123` | Secret code to change protected admin password |
| `PYTHON_BIN` | No | *(auto-detected)* | Explicit path to Python binary |
| `ENVIRONMENT_FILES_DIR` | No | `./environment_files` | Root directory for game files on disk |

---

## Project Structure

```
arc-agi-internal/
├── package.json                  # Monorepo root — all deps (client + server)
├── tsconfig.json                 # Shared TypeScript config
├── vite.config.ts                # Vite config (root: client/, build → dist/public/)
├── tailwind.config.ts            # Tailwind CSS config
├── postcss.config.js             # PostCSS (autoprefixer + tailwind)
├── .env                          # Environment variables (gitignored)
├── .python-bin                   # Auto-written by postinstall: path to Python binary
├── .gitignore
│
├── client/                       # React SPA
│   ├── index.html                # Vite entry HTML
│   └── src/
│       ├── main.tsx              # React root mount
│       ├── App.tsx               # Router: public routes + admin routes + guards
│       ├── api/
│       │   └── client.ts         # Axios client, all API functions (auth/games/player/analytics/users/requests)
│       ├── hooks/
│       │   ├── useAuth.tsx       # Auth context provider + useAuth hook
│       │   ├── useGameEngine.tsx # Game session state + action dispatcher
│       │   └── useVideoRecorder.ts  # Screen capture via getDisplayMedia
│       ├── components/
│       │   ├── Layout.tsx        # Admin shell (sidebar + outlet)
│       │   ├── GameCanvas.tsx    # Canvas renderer for ARC grids (click + keyboard)
│       │   ├── GamePreviewCanvas.tsx  # Thumbnail canvas for homepage cards
│       │   ├── GameUploadForm.tsx     # Upload form (game.py + metadata.json + metadata fields)
│       │   ├── VideoRecorder.tsx      # Record prompt, controls, preview modal, upload
│       │   └── ConfirmModal.tsx       # Reusable confirm/cancel dialog
│       ├── pages/
│       │   ├── HomePage.tsx           # Public: game grid, stats, tabs (Games | Play Your Own | Request Upload)
│       │   ├── PublicPlayPage.tsx      # Public: /play/:gameId — full game player
│       │   ├── DirectPlayPage.tsx     # Public: /play-direct — ephemeral upload-and-play
│       │   ├── LoginPage.tsx          # Admin: /admin/login
│       │   ├── DashboardPage.tsx      # Admin: /admin — stats cards, charts, recent games
│       │   ├── GamesPage.tsx          # Admin: /admin/games — game table, toggle, delete, sync
│       │   ├── GameDetailPage.tsx     # Admin: /admin/games/:gameId — tabs: overview, analytics, source, sessions, videos
│       │   ├── GameUploadPage.tsx     # Admin: /admin/games/upload — upload new game
│       │   ├── GamePlayPage.tsx       # Admin: /admin/games/:gameId/play — play game (authenticated)
│       │   ├── RequestedGamesPage.tsx # Admin: /admin/requests — pending/approved/rejected requests
│       │   ├── TempGamesPage.tsx      # Admin: /admin/temp-games — ephemeral session log
│       │   └── UsersPage.tsx          # Admin: /admin/users — create/edit/delete users
│       └── types/
│           └── index.ts               # Client-side type re-exports
│
├── server/                       # Express backend
│   ├── index.ts                  # Entry point: middleware, routes, Vite/static setup, listen
│   ├── db.ts                     # PostgreSQL pool + initDB (table creation) + helpers
│   ├── routes.ts                 # Route registration: health check + 6 routers
│   ├── vite.ts                   # Vite dev middleware integration + SPA fallback
│   ├── middleware/
│   │   ├── auth.ts               # JWT creation/verification, bcrypt, ensureDefaultAdmin, requireAdmin, requirePage
│   │   ├── asyncHandler.ts       # Wraps async route handlers for error propagation
│   │   └── errorHandler.ts       # AppError class + Express error handler
│   ├── routes/
│   │   ├── auth.ts               # POST /login, GET /me, POST /register
│   │   ├── games.ts              # 7 public + 13 admin endpoints (CRUD, files, toggle, source, sync, videos)
│   │   ├── player.ts             # 3 public + 3 ephemeral + 4 admin + 1 palette endpoint
│   │   ├── analytics.ts          # 4 auth + 5 admin endpoints (dashboard, game stats, sessions, export)
│   │   ├── users.ts              # 4 admin endpoints + 1 protected password change
│   │   └── requests.ts           # 1 public + 5 admin endpoints (submit, list, review, source, files, delete)
│   ├── services/
│   │   └── GamePythonBridge.ts   # Python subprocess manager (spawn, NDJSON, timeout, kill)
│   └── python/
│       └── game_runner.py        # Python NDJSON bridge: init, action, reset, quit
│
├── shared/
│   └── types.ts                  # Shared TypeScript types (User, Game, PlaySession, GameFrame, LevelStat, etc.)
│
├── scripts/
│   └── setup-python.cjs          # Postinstall: detect Python binary, install arc-agi, write .python-bin
│
└── environment_files/            # Game files on disk (gitignored contents vary)
    ├── <game_code>/
    │   └── <version>/
    │       ├── <game_code>.py
    │       ├── metadata.json
    │       └── recordings/       # Uploaded gameplay recordings (.webm)
    └── _requests/                # Pending request uploads (cleaned up on approve/reject)
        └── <game_id>/
            ├── <game_code>.py
            └── metadata.json
```

---

## System Flows

### Flow 1: Public User Plays a Game

1. User opens the homepage (`/`), which fetches `GET /api/games/public` to display active games as a grid of preview cards.
2. User clicks a game card. A name modal appears prompting for an optional player name.
3. User enters name and clicks Play. Browser navigates to `/play/:gameId?name=<name>`.
4. `PublicPlayPage` mounts and calls `POST /api/player/public/start` with `{game_id, player_name}`.
5. Server looks up the game in the DB, spawns a `GamePythonBridge` subprocess, calls `bridge.init()` which loads the game `.py` file via `importlib`, instantiates the game class, and performs an initial `RESET` action.
6. A `play_sessions` row is inserted. The initial frame (grid, width, height, state, level, available_actions) is returned to the client.
7. A `SessionTracker` is initialized server-side to track per-level timing, actions, game_overs, and resets.
8. The `GameCanvas` component renders the grid. The player uses keyboard (WASD/Arrow keys/Space) or clicks grid cells to send actions.
9. Each action calls `POST /api/player/public/action/:sessionGuid` with `{action, x?, y?}`. The server forwards the action to the Python bridge, receives the updated frame, detects level transitions and game_over events, updates `SessionTracker` counters, persists to `play_sessions`, and returns the frame.
10. On level clear (level number increases), the completed level's stats (actions, time, game_overs, resets) are recorded in `completedLevels`. The client shows a glitch-effect level clear animation.
11. On WIN or GAME_OVER terminal state, the session's `ended_at` is set, `games.total_plays` is incremented (and `total_wins` on WIN). The client shows a game over/win overlay with final stats.
12. When the user leaves the page, `POST /api/player/public/end/:sessionGuid` is called, which kills the Python bridge, cleans up in-memory state, and marks unfinished sessions as GAME_OVER.

### Flow 2: Game Request Submission & Approval

1. Public user opens the homepage and clicks the "Request Upload" tab.
2. Fills out the form: requester name, email, description, game rules, owner name, drive link, video link, and uploads `game.py` + `metadata.json`.
3. Submits via `POST /api/requests/submit` (multipart). Server validates metadata JSON, checks `game_id` uniqueness against both `games` and pending `game_requests`, validates URLs (SSRF protection: blocks localhost, private IPs, verifies with HEAD request). Files are saved to `environment_files/_requests/<game_id>/` and binary content is also stored as BYTEA in the `game_requests` table.
4. Admin navigates to `/admin/requests`, sees the pending request with all metadata.
5. Admin can view the submitted source code (`GET /api/requests/:id/source`) and download files (`GET /api/requests/:id/files/game` or `/metadata`).
6. Admin clicks Approve → `POST /api/requests/:id/review` with `{action: "approve"}`. Server copies files from `_requests/` to `environment_files/<game_code>/<version>/`, creates a `games` row with `is_active = false`, and deletes the request.
7. Admin navigates to `/admin/games`, finds the new game (inactive), and toggles it active via `PATCH /api/games/:gameId/toggle`.
8. Game now appears on the public homepage.

### Flow 3: Direct Play (Ephemeral)

1. User opens the homepage and clicks the "Play Your Own" tab, or navigates to `/play-direct`.
2. Uploads `game.py` + `metadata.json` and optionally enters a player name.
3. `POST /api/player/ephemeral/start` creates a temp directory in the OS temp folder, writes the uploaded files, spawns a `GamePythonBridge`, and inserts a row into `temp_game_sessions` with a 24-hour `expires_at`.
4. Session GUID is prefixed with `eph_` to distinguish ephemeral sessions.
5. Actions go through `POST /api/player/ephemeral/action/:sessionGuid`, updating `temp_game_sessions` including an `action_log` array.
6. On end (`POST /api/player/ephemeral/end/:sessionGuid`), the Python bridge is killed, the temp directory is deleted, and the session is marked as GAME_OVER.
7. No permanent `games` row is created. Sessions remain in `temp_game_sessions` until an admin deletes them.

### Flow 4: Video Recording

1. When a game starts playing, the `VideoRecorder` component shows a prompt asking the user to Record or Skip.
2. If the user clicks Record, `getDisplayMedia()` requests screen capture permission. A `MediaRecorder` is started using the best available codec (MP4 H.264 preferred, WebM VP9/VP8 fallback) at 8 Mbps.
3. Recording time is displayed in the UI. Recording auto-stops when the game reaches a terminal state (WIN or GAME_OVER).
4. A preview modal shows the recorded video. The user can download locally or push to the server.
5. Server upload goes to `POST /api/games/public/:gameId/video` (multipart), which saves the `.webm` file to `environment_files/<game_code>/<version>/recordings/<player>_<timestamp>.webm`.
6. Admins can view, stream, and delete recordings from the Game Detail page's Videos tab.

### Flow 5: Admin Game Management

1. **Upload**: `/admin/games/upload` — upload `game.py` + `metadata.json` with optional metadata fields. Server validates, writes to `environment_files/`, creates DB record with `is_active = false`.
2. **Edit metadata**: `PUT /api/games/:gameId` — update name, description, rules, owner, links, fps, tags, active status.
3. **Replace files**: `PUT /api/games/:gameId/files` — upload new `game.py` and/or `metadata.json` to replace existing files on disk.
4. **Toggle active**: `PATCH /api/games/:gameId/toggle` — show/hide from public homepage.
5. **View source**: `GET /api/games/:gameId/source` — returns game Python source and parsed metadata JSON.
6. **Sync from filesystem**: `POST /api/games/sync-local` — scans `environment_files/` for directories with `metadata.json` not yet in the DB and creates game records.
7. **View sessions**: Analytics tab shows per-game stats; Sessions tab shows per-player cards with level-by-level breakdown.
8. **Export**: `GET /api/analytics/export/:gameId` — downloads `.xlsx` with session data + summary sheet. Supports date filters.
9. **Manage recordings**: List, stream, and delete `.webm` recordings.

### Flow 6: Reset Behavior

The RESET action (keyboard `R`) has context-dependent behavior tracked by `SessionTracker.actionsSinceReset`:

| Condition | Press R | Result |
|-----------|---------|--------|
| Player has made moves on current level (`actionsSinceReset > 0`) | 1st R | Reset same level. Level timer restarts. `levelResets++`. `actionsSinceReset` resets to 0. |
| Player has NOT moved since last reset (`actionsSinceReset === 0`) | 2nd R (double reset) | Full restart to Level 1. `completedLevels` cleared. Session `startTime` resets. All per-level counters reset. |
| After GAME_OVER | RESET | Continues same level. `levelGameOvers` was already incremented on the GAME_OVER transition. |

---

## Python Bridge Architecture

### Subprocess Lifecycle

`GamePythonBridge` (`server/services/GamePythonBridge.ts`) manages one Python child process per active game session:

1. **Spawn**: `spawn(pythonBin, ["server/python/game_runner.py"])` with `PYTHONUNBUFFERED=1`.
2. **Communication**: Strictly request-response over NDJSON. One JSON command written to stdin, one JSON response read from stdout. At most one request is in-flight at any time.
3. **Timeout**: Each command has a 30-second timeout.
4. **Cleanup**: `bridge.kill()` sends a `quit` command, then `SIGTERM`. The readline interface is closed.

### NDJSON Protocol

**Commands (Node → Python via stdin):**

| Command | Payload | Description |
|---------|---------|-------------|
| `init` | `{command, game_id, game_path, seed?}` | Load game file, instantiate class, return initial frame |
| `action` | `{command, action, x?, y?}` | Execute game action (ACTION1-7, RESET). ACTION6 supports x/y coordinates |
| `reset` | `{command}` | Reset game to initial state, return frame |
| `quit` | `{command}` | Exit cleanly |

**Responses (Python → Node via stdout):**

| Type | Fields | When |
|------|--------|------|
| `ready` | `{type, game_id, frame, metadata}` | After successful `init` |
| `frame` | `{type, grid, width, height, state, level, total_actions, available_actions, ...}` | After `action` or `reset` |
| `error` | `{type, code, message}` | On any error |
| `quit` | `{type, message}` | Acknowledging `quit` |

### Python Binary Detection

Resolution order (in `GamePythonBridge.resolvePythonBin()`):

1. `PYTHON_BIN` environment variable
2. `.python-bin` marker file (written by `scripts/setup-python.cjs` during `npm install`)
3. `python3` on Unix / `python` on Windows

The postinstall script (`scripts/setup-python.cjs`) tries these candidates in order: project `.venv/bin/python3`, Homebrew Python, system Python. It checks for `arcengine` import, and if not found, attempts `pip install arc-agi`.

### Game Loading

`game_runner.py` loads game files via `importlib.util.spec_from_file_location`. The module is registered in `sys.modules` before `exec_module` so that `@dataclass` decorators (which look up the module by name) work correctly. The first `ARCBaseGame` subclass found in the module is instantiated.

### Animation Frames

The ARCEngine returns `frame_data.frame` as a list of animation frames. On level clear, this list contains `[victory_frame, new_level_frame]`. The bridge always takes the **last** frame (`frame_list[-1]`), which represents the current playable game state rather than a transient victory animation.

---

## Database Schema

All tables are created automatically on server start via `initDB()` in `server/db.ts`.

### `users`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `username` | `VARCHAR(50)` | UNIQUE NOT NULL | Login name |
| `hashed_password` | `VARCHAR(255)` | NOT NULL | bcrypt hash |
| `email` | `VARCHAR(255)` | UNIQUE | Optional email |
| `is_admin` | `BOOLEAN` | DEFAULT FALSE | Admin flag |
| `is_active` | `BOOLEAN` | DEFAULT TRUE | Account active flag |
| `allowed_pages` | `JSONB` | DEFAULT '[]' | Page-level access control list |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `updated_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |

### `games`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `game_id` | `VARCHAR(50)` | UNIQUE NOT NULL | Human-readable ID (e.g. `fm01`, `ls20-v2`) |
| `name` | `VARCHAR(100)` | NOT NULL | Display name |
| `description` | `TEXT` | | Game description |
| `game_rules` | `TEXT` | | Rules text |
| `game_owner_name` | `VARCHAR(100)` | | Game creator name |
| `game_drive_link` | `VARCHAR(500)` | | External link |
| `game_video_link` | `VARCHAR(500)` | | External video link |
| `version` | `VARCHAR(20)` | NOT NULL DEFAULT 'v1' | Version string |
| `game_code` | `VARCHAR(10)` | NOT NULL | Base code (game_id without version suffix) |
| `is_active` | `BOOLEAN` | DEFAULT TRUE | Visible on public homepage |
| `default_fps` | `INTEGER` | DEFAULT 5 | Animation FPS |
| `baseline_actions` | `JSONB` | | Array of baseline action counts per level |
| `tags` | `JSONB` | | String array of tags |
| `grid_max_size` | `INTEGER` | DEFAULT 64 | Max grid dimension |
| `game_file_path` | `VARCHAR(500)` | NOT NULL | Absolute path to `.py` file |
| `metadata_file_path` | `VARCHAR(500)` | NOT NULL | Absolute path to `metadata.json` |
| `local_dir` | `VARCHAR(500)` | NOT NULL | Absolute path to game directory |
| `total_plays` | `INTEGER` | DEFAULT 0 | Aggregate play count |
| `total_wins` | `INTEGER` | DEFAULT 0 | Aggregate win count |
| `avg_score` | `REAL` | DEFAULT 0.0 | Average score |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `updated_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `uploaded_by` | `TEXT` | FK → users(id) | Admin who uploaded |

### `play_sessions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `game_id` | `TEXT` | FK → games(id) ON DELETE CASCADE | |
| `user_id` | `TEXT` | FK → users(id) | NULL for public players |
| `session_guid` | `VARCHAR(100)` | UNIQUE NOT NULL | Session identifier |
| `seed` | `INTEGER` | DEFAULT 0 | Game seed |
| `current_level` | `INTEGER` | DEFAULT 0 | Last level reached (0-indexed) |
| `state` | `VARCHAR(20)` | DEFAULT 'NOT_FINISHED' | WIN, GAME_OVER, NOT_FINISHED |
| `score` | `REAL` | DEFAULT 0.0 | |
| `total_actions` | `INTEGER` | DEFAULT 0 | |
| `game_overs` | `INTEGER` | DEFAULT 0 | Total deaths across all levels |
| `player_name` | `VARCHAR(100)` | | Display name |
| `total_time` | `REAL` | DEFAULT 0.0 | Elapsed seconds |
| `level_stats` | `JSONB` | DEFAULT '[]' | Array of per-level stat objects |
| `action_log` | `JSONB` | DEFAULT '[]' | Full action history |
| `started_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `ended_at` | `TIMESTAMPTZ` | | Set on terminal state |

### `game_analytics`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `game_id` | `TEXT` | FK → games(id) ON DELETE CASCADE | |
| `date` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `plays_count` | `INTEGER` | DEFAULT 0 | |
| `wins_count` | `INTEGER` | DEFAULT 0 | |
| `avg_actions_to_win` | `REAL` | DEFAULT 0.0 | |
| `avg_score` | `REAL` | DEFAULT 0.0 | |
| `unique_players` | `INTEGER` | DEFAULT 0 | |

### `game_requests`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `game_id` | `VARCHAR(50)` | UNIQUE NOT NULL | Requested game ID |
| `requester_name` | `VARCHAR(100)` | NOT NULL | Submitter name |
| `requester_email` | `VARCHAR(255)` | | |
| `message` | `TEXT` | | Submission note |
| `description` | `TEXT` | | Game description |
| `game_rules` | `TEXT` | | |
| `game_owner_name` | `VARCHAR(100)` | | |
| `game_drive_link` | `VARCHAR(500)` | | URL validated (SSRF-safe) |
| `game_video_link` | `VARCHAR(500)` | | URL validated (SSRF-safe) |
| `status` | `VARCHAR(20)` | DEFAULT 'pending' | pending, approved, rejected |
| `admin_note` | `TEXT` | | Admin review note |
| `game_file_path` | `VARCHAR(500)` | | Path in `_requests/` |
| `metadata_file_path` | `VARCHAR(500)` | | |
| `local_dir` | `VARCHAR(500)` | | |
| `game_file_content` | `BYTEA` | | Binary backup of .py |
| `metadata_file_content` | `BYTEA` | | Binary backup of metadata |
| `game_code` | `VARCHAR(50)` | | |
| `version` | `VARCHAR(20)` | DEFAULT 'v1' | |
| `tags` | `JSONB` | | |
| `default_fps` | `INTEGER` | DEFAULT 5 | |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `reviewed_at` | `TIMESTAMPTZ` | | |
| `reviewed_by` | `TEXT` | FK → users(id) | |

### `temp_game_sessions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `session_guid` | `VARCHAR(50)` | UNIQUE NOT NULL | Prefixed with `eph_` |
| `game_id` | `VARCHAR(50)` | | Game ID string (not FK) |
| `player_name` | `VARCHAR(100)` | | |
| `state` | `VARCHAR(20)` | DEFAULT 'NOT_FINISHED' | |
| `score` | `REAL` | DEFAULT 0.0 | |
| `total_actions` | `INTEGER` | DEFAULT 0 | |
| `current_level` | `INTEGER` | DEFAULT 0 | |
| `game_overs` | `INTEGER` | DEFAULT 0 | |
| `total_time` | `REAL` | DEFAULT 0.0 | |
| `level_stats` | `JSONB` | | |
| `action_log` | `JSONB` | DEFAULT '[]' | |
| `started_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `ended_at` | `TIMESTAMPTZ` | | |
| `expires_at` | `TIMESTAMPTZ` | NOT NULL | Set to `NOW() + 1 day` |

### Indexes

```sql
CREATE INDEX idx_games_game_id    ON games(game_id);
CREATE INDEX idx_games_active     ON games(is_active);
CREATE INDEX idx_sessions_game    ON play_sessions(game_id);
CREATE INDEX idx_sessions_guid    ON play_sessions(session_guid);
CREATE INDEX idx_requests_status  ON game_requests(status);
CREATE INDEX idx_temp_guid        ON temp_game_sessions(session_guid);
```

---

## API Reference

All routes are prefixed with `/api`. Authentication is via `Authorization: Bearer <jwt>` header.

### Health Check

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/health` | None | Returns `{status: "ok"}` |

### Auth — `/api/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | None | Login with `{username, password}`, returns JWT |
| `GET` | `/auth/me` | JWT | Get current user profile |
| `POST` | `/auth/register` | Admin | Create new user account |

### Games — `/api/games`

**Public (no auth):**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/games/public` | None | List all active games |
| `GET` | `/games/public/stats` | None | Homepage aggregate stats + top games |
| `GET` | `/games/public/:gameId` | None | Get single active game details |
| `GET` | `/games/public/:gameId/plays` | None | Recent play sessions (leaderboard) |
| `GET` | `/games/public/:gameId/stats` | None | Per-game stats (plays, wins, top performer) |
| `GET` | `/games/public/:gameId/preview` | None | Initial grid frame for thumbnail rendering |
| `POST` | `/games/public/:gameId/video` | None | Upload gameplay recording (multipart, `video` field) |

**Admin (JWT + admin):**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/games/` | Admin | List all games (optional `?active_only=true`) |
| `GET` | `/games/:gameId` | Admin | Get game details (including inactive) |
| `POST` | `/games/upload` | Admin | Upload new game (multipart: `game_file` + `metadata_file` + form fields) |
| `PUT` | `/games/:gameId` | Admin | Update game metadata |
| `PUT` | `/games/:gameId/files` | Admin | Replace game files on disk |
| `PATCH` | `/games/:gameId/toggle` | Admin | Toggle `is_active` |
| `DELETE` | `/games/:gameId` | Admin | Delete game + sessions + files |
| `DELETE` | `/games/:gameId/sessions` | Admin | Clear sessions + reset counters |
| `GET` | `/games/:gameId/source` | Admin | View `.py` source + parsed metadata |
| `POST` | `/games/sync-local` | Admin | Scan filesystem, create DB records for new games |
| `GET` | `/games/:gameId/videos` | Admin | List recordings |
| `GET` | `/games/:gameId/videos/:filename` | Admin | Stream recording file |
| `DELETE` | `/games/:gameId/videos/:filename` | Admin | Delete recording |

### Player — `/api/player`

**Public (no auth):**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/player/public/start` | None | Start public session: `{game_id, seed?, player_name?, start_level?}` |
| `POST` | `/player/public/action/:sessionGuid` | None | Send action: `{action, x?, y?}` |
| `POST` | `/player/public/end/:sessionGuid` | None | End session, kill bridge |

**Ephemeral (no auth):**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/player/ephemeral/start` | None | Upload files + start temp session (multipart) |
| `POST` | `/player/ephemeral/action/:sessionGuid` | None | Send action in ephemeral session |
| `POST` | `/player/ephemeral/end/:sessionGuid` | None | End + cleanup temp files |

**Authenticated:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/player/start` | JWT | Start authenticated session |
| `POST` | `/player/action/:sessionGuid` | JWT | Send action (authenticated) |
| `GET` | `/player/frame/:sessionGuid` | JWT | Get current frame without action |
| `POST` | `/player/end/:sessionGuid` | JWT | End authenticated session |

**Utility:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/player/palette` | None | ARC color palette as JSON map |

### Analytics — `/api/analytics`

**Authenticated:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/analytics/dashboard` | JWT | Dashboard stats: counts, daily plays/wins, game distribution |
| `GET` | `/analytics/game/:gameId` | JWT | Per-game analytics (optional `?days=30`) |
| `GET` | `/analytics/sessions` | JWT | Recent sessions (optional `?game_id=&limit=20`) |
| `GET` | `/analytics/replay/:sessionId` | JWT | Full session replay data with action_log |

**Admin:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/analytics/temp-sessions` | Admin | List all ephemeral sessions |
| `DELETE` | `/analytics/temp-sessions` | Admin | Delete all ephemeral sessions |
| `GET` | `/analytics/export/:gameId` | Admin | Export sessions as .xlsx (filters: `?filter=all\|today\|date\|range`) |
| `GET` | `/analytics/export-all` | Admin | Export all games' sessions as .xlsx |
| `GET` | `/analytics/export-games` | Admin | Export games list as .xlsx |

### Users — `/api/users`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/users/` | Admin | List all users |
| `GET` | `/users/:userId` | Admin | Get single user |
| `PUT` | `/users/:userId` | Admin | Update user (email, is_admin, is_active, password, allowed_pages) |
| `DELETE` | `/users/:userId` | Admin | Delete user (cannot delete self or protected user) |
| `POST` | `/users/protected/change-password` | JWT | Protected admin password change (requires `secret_code`) |

### Requests — `/api/requests`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/requests/submit` | None | Submit game for review (multipart + form fields) |
| `GET` | `/requests/` | Admin | List requests by status (`?status=pending\|approved\|rejected\|all`) |
| `GET` | `/requests/:requestId` | Admin | Get request details |
| `GET` | `/requests/:requestId/source` | Admin | View submitted source code + metadata |
| `GET` | `/requests/:requestId/files/:fileType` | Admin | Download submitted file (`game` or `metadata`) |
| `POST` | `/requests/:requestId/review` | Admin | Approve or reject: `{action: "approve"\|"reject", admin_note?}` |
| `DELETE` | `/requests/:requestId` | Admin | Delete request + cleanup files |

### Per-Level Stats JSON Structure

Stored in `play_sessions.level_stats` and `temp_game_sessions.level_stats`:

```json
[
  {
    "level": 0,
    "actions": 24,
    "time": 45.3,
    "completed": true,
    "game_overs": 3,
    "resets": 1
  },
  {
    "level": 1,
    "actions": 12,
    "time": 22.1,
    "completed": true,
    "game_overs": 0,
    "resets": 0
  }
]
```

---

## Game File Format

Each game lives in `environment_files/<game_code>/<version>/` and consists of two files.

### metadata.json

```json
{
  "game_id": "fm01",
  "default_fps": 5,
  "baseline_actions": [12, 16, 20, 24, 28, 32],
  "tags": ["mathematics", "fractions", "visual_reasoning"],
  "local_dir": "environment_files/fm01/v1"
}
```

**Allowed fields** (server rejects unknown keys):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `game_id` | `string` | **Yes** | Must match `/^[a-z]{2,6}\d{1,4}$/` or `/^[a-z0-9]{2,10}-v\d+$/` |
| `default_fps` | `integer` | No | Animation speed (default 5) |
| `baseline_actions` | `integer[]` | No | Baseline action counts per level |
| `tags` | `string[]` | No | Descriptive tags |
| `local_dir` | `string` | No | Auto-set on upload |

**game_id format rules:**
- Basic: 2-6 lowercase letters + 1-4 digits (e.g. `fm01`, `ls20`, `abc1234`)
- Versioned: 2-10 lowercase alphanumeric + `-v` + digits (e.g. `fm01-v2`, `test123-v1`)

### game.py

Must contain a class that extends `ARCBaseGame` from the `arcengine` package:

```python
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

class MyGame(ARCBaseGame):
    def __init__(self, seed=0):
        camera = Camera(background=5, width=9, height=9)
        super().__init__(game_id="mygame01", levels=levels, camera=camera)

    def step(self):
        # Handle actions: self.action.id is a GameAction enum
        if self.action.id == GameAction.ACTION1:
            # ... game logic
            pass
        self.complete_action()
```

The Python bridge finds the first `ARCBaseGame` subclass in the module and instantiates it. Actions are `RESET`, `ACTION1` through `ACTION7`. `ACTION6` supports `(x, y)` coordinates for grid click interactions.

---

## Color Palette

The ARC-AGI-3 palette uses 16 indexed colors (defined in `shared/types.ts` and `client/src/hooks/useGameEngine.tsx`):

| Index | Name | Hex |
|-------|------|-----|
| 0 | White | `#FFFFFF` |
| 1 | Off-white | `#CCCCCC` |
| 2 | Light Grey | `#999999` |
| 3 | Grey | `#666666` |
| 4 | Dark Grey | `#333333` |
| 5 | Black | `#000000` |
| 6 | Magenta | `#E53AA3` |
| 7 | Pink | `#FF7BCC` |
| 8 | Red | `#F93C31` |
| 9 | Blue | `#1E93FF` |
| 10 | Light Blue | `#88D8F1` |
| 11 | Yellow | `#FFDC00` |
| 12 | Orange | `#FF851B` |
| 13 | Maroon | `#921231` |
| 14 | Green | `#4FCC30` |
| 15 | Purple | `#A356D6` |

---

## Frontend Pages

| Route | Component | Auth | Description |
|-------|-----------|------|-------------|
| `/` | `HomePage` | None | Public homepage — game grid, "Play Your Own" tab, "Request Upload" tab, aggregate stats |
| `/play/:gameId` | `PublicPlayPage` | None | Full game player with canvas, keyboard controls, timer, level stats, video recorder |
| `/play-direct` | `DirectPlayPage` | None | Ephemeral upload-and-play |
| `/admin/login` | `LoginPage` | None | Admin login form |
| `/admin` | `DashboardPage` | JWT | Dashboard — stat cards, 7-day charts (recharts), game distribution, recent games, top played |
| `/admin/games` | `GamesPage` | JWT + `games` | Game management table — search, filter, toggle active, delete, sync from filesystem |
| `/admin/games/upload` | `GameUploadPage` | JWT + `upload` | Upload new game form |
| `/admin/games/:gameId` | `GameDetailPage` | JWT + `games` | Game detail — tabs: Overview, Analytics, Source Code, Sessions (with level breakdown), Videos |
| `/admin/games/:gameId/play` | `GamePlayPage` | JWT + `games` | Play game as authenticated admin |
| `/admin/requests` | `RequestedGamesPage` | JWT + `requests` | Game request management — view, approve/reject, view source |
| `/admin/temp-games` | `TempGamesPage` | JWT + `games` | Ephemeral session log — view and bulk delete |
| `/admin/users` | `UsersPage` | JWT + `users` | User management — create, edit roles/pages, delete |

---

## Authentication & Authorization

### JWT Flow

1. Client sends `POST /api/auth/login` with `{username, password}`.
2. Server verifies bcrypt hash, returns `{access_token, token_type: "bearer"}`.
3. Client stores token in `localStorage` as `arc_token`.
4. All subsequent requests include `Authorization: Bearer <token>` via Axios interceptor.
5. On 401 response while on `/admin/*`, client clears token and redirects to login.

### Role-Based Access

- **Admin** (`is_admin: true`): Full access to all admin routes and pages.
- **Regular user** (`is_admin: false`): Access controlled by `allowed_pages` array. Pages: `dashboard`, `games`, `upload`, `requests`, `users`.
- **Public**: No auth required for homepage, play pages, game request submission, and ephemeral play.

### Page-Level Access Control

- `allowed_pages` is a JSONB array on the `users` table.
- `PageGuard` component in `App.tsx` checks `user.allowed_pages.includes(page)` before rendering.
- Admins bypass all page guards.
- Server-side: `requireAdmin` middleware enforces admin-only endpoints. `requirePage(page)` middleware is available but admin routes currently use `requireAdmin`.

### Protected Super Admin

If `PROTECTED_USERNAME` is set in `.env`, a super admin account is created on first start. This account:
- Cannot be modified or deleted by other admins via the Users API.
- Password can only be changed via `POST /api/users/protected/change-password` with the correct `PROTECTED_SECRET_CODE`.

---

## Build & Deploy

### Development

```bash
npm run dev
```

Runs `cross-env NODE_ENV=development node --import tsx server/index.ts`. Vite dev server is mounted as Express middleware with HMR over the same HTTP server. Single port (default 5000).

### Production Build

```bash
npm run build
```

Two steps:
1. `vite build` — compiles React app to `dist/public/` (with asset hashing).
2. `esbuild server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist` — bundles server to `dist/index.js`.

### Production Start

```bash
npm start
```

Runs `cross-env NODE_ENV=production node dist/index.js`. Serves static files from `dist/public/` and handles SPA fallback for non-API routes.

### Type Check

```bash
npm run check
```

Runs `tsc --noEmit` against the full project.

### Docker Considerations

- Ensure Python 3.10+ and `arc-agi` are available in the container.
- PostgreSQL must be reachable (use `DB_HOST`, `DB_PORT`, etc.).
- `environment_files/` should be a persistent volume so game files and recordings survive container restarts.
- The `npm install` postinstall script will attempt to auto-install `arc-agi` via pip.
- Set `SECRET_KEY` to a strong random value.
- Set `NODE_ENV=production`.

---

## Troubleshooting

### Python not found

```
[GamePythonBridge] Failed to spawn Python: ...
```

- Check that Python 3.10+ is installed: `python3 --version`
- Check that `arcengine` is importable: `python3 -c "import arcengine; print('ok')"`
- If using a venv, ensure `.python-bin` points to the correct binary, or set `PYTHON_BIN` in `.env`.
- Re-run `npm install` to trigger the postinstall detection script.

### Database connection refused

```
[DB] Unexpected pool error: connect ECONNREFUSED 127.0.0.1:5432
```

- Ensure PostgreSQL is running: `pg_isready`
- Check `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` in `.env`.
- The database must exist before starting: `createdb arc_agi_db`

### Port already in use

```
Error: listen EADDRINUSE: address already in use 0.0.0.0:5000
```

- Kill the existing process: `lsof -ti:5000 | xargs kill`
- Or change the port via `PORT` env var.

### JSONB serialization errors

If you see `invalid input syntax for type json`, ensure JSONB values passed to queries go through `toJsonb()` (from `server/db.ts`), which handles `null`, `undefined`, and already-stringified values.

### Game file validation failures

- `metadata.json` must contain `game_id` and only allowed fields: `game_id`, `default_fps`, `baseline_actions`, `tags`, `local_dir`.
- `game_id` must match `/^[a-z]{2,6}\d{1,4}$/` or `/^[a-z0-9]{2,10}-v\d+$/`.
- The `.py` file must contain a class that extends `ARCBaseGame`.
- `game_id` must be unique across both `games` and pending `game_requests`.

### Game subprocess timeout

```
Command timed out after 30000ms: action
```

The Python bridge has a 30-second timeout per command. This usually indicates the game's `step()` method is hanging or has an infinite loop. Check the game `.py` source code.

### Recordings not saving

- The `recordings/` subdirectory is created automatically under the game's `local_dir`.
- Ensure the Node process has write permissions to `environment_files/`.
- Video uploads are limited by Express body size (`50mb` limit configured in `server/index.ts`).
