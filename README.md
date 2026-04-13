# ARC-AGI-3 Internal Platform

Internal full-stack platform for managing, playtesting, and analyzing ARC-AGI-3 grid-based puzzle games. Features a retro arcade-themed public homepage where players can browse and play active games, an ephemeral "Play Your Own" mode for testing uploaded game files, a role-based dashboard with 4 user roles (Tasker, QL, PL, Super Admin), a multi-stage game approval workflow, team management with many-to-many lead assignments, real-time SSE notifications with sound, bulk file downloads, per-level session tracking, gameplay stats, audit trails, and optional screen recording with server upload. Built as a single-port Node.js/Express + React + Vite monorepo, with Python running only as a subprocess for the game engine.

---

## Architecture

```
Single-port monorepo: Express (API + static) + Vite (React HMR) + Python subprocess

┌──────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  React SPA (Vite + React Router + Tailwind + Framer Motion)  │
│  Arcade-themed public pages + Role-based dashboard           │
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
│  │  (9 routers)   │  │  + HMR (dev) │  │  dist/public/  │  │
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
│  │  10 tables    │  │  <game_code>/game.py                │  │
│  │              │  │  <game_code>/metadata.json           │  │
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
- **PostgreSQL 14+** running remotely
- **Python 3.12+** with the `arc-agi` package (provides `arcengine`)

### Setup

```bash
git clone https://github.com/sarbeshtiwari/arc-agi-internal.git
cd arc-agi-internal
npm install        # also runs postinstall → auto-detects Python, installs arc-agi
npm run dev        # starts Express + Vite on http://localhost:5000
```

On first start:
1. All 10 database tables are created automatically via `initDB()`.
2. A default admin user is created from `DEFAULT_ADMIN_USERNAME` / `DEFAULT_ADMIN_PASSWORD` env vars (defaults: `admin` / `admin123`) with role `super_admin`.
3. If `PROTECTED_USERNAME` is set, a protected super admin is also created.
4. Existing users/games are migrated (adds `role`, `display_name`, `approval_status` columns if missing, creates `user_team_leads` junction table).

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
| `SECRET_KEY` | **Yes** | `-` | JWT signing secret (change in production) |
| `JWT_EXPIRES_IN` | No | `1440m` | JWT token expiry (24 hours) |
| `PYTHON_BIN` | No | *(auto-detected)* | Explicit path to Python binary |
| `ENVIRONMENT_FILES_DIR` | No | `./environment_files` | Root directory for game files on disk |

---

## Project Structure

```
arc-agi-internal/
├── package.json                  # Monorepo root — all deps (client + server)
├── tsconfig.json                 # Shared TypeScript config
├── vite.config.ts                # Vite config (root: client/, build → dist/public/)
├── tailwind.config.ts            # Tailwind CSS config (pixel font, neon/cabinet colors, arcade animations)
├── postcss.config.js             # PostCSS (autoprefixer + tailwind)
├── .env                          # Environment variables (gitignored)
├── .python-bin                   # Auto-written by postinstall: path to Python binary
├── .gitignore
│
├── client/                       # React SPA
│   ├── index.html                # Vite entry HTML (loads Press Start 2P pixel font)
│   └── src/
│       ├── main.tsx              # React root mount
│       ├── App.tsx               # Router: public routes + /dashboard/* (role-based) + guards
│       ├── index.css             # Tailwind imports + arcade.css
│       ├── arcade.css            # CRT screen, scanlines, cabinet, neon utilities, starfield
│       ├── api/
│       │   └── client.ts         # Axios client, all API modules (auth/games/player/analytics/users/requests/approval/teams/notifications)
│       ├── hooks/
│       │   ├── useAuth.tsx       # Auth context: login/logout, role helpers (isSuperAdmin/isQL/isPL/isTasker)
│       │   ├── useGameEngine.tsx # Game session state + action dispatcher
│       │   └── useVideoRecorder.ts  # Server-streaming screen capture (zero client RAM)
│       ├── components/
│       │   ├── Layout.tsx        # Legacy admin shell (unused, redirects to /dashboard)
│       │   ├── DashboardLayout.tsx   # Role-aware dashboard sidebar + NotificationBell
│       │   ├── NotificationBell.tsx  # Real-time SSE notifications + Web Audio sound
│       │   ├── GameCanvas.tsx    # Canvas renderer for ARC grids (click + keyboard)
│       │   ├── GamePreviewCanvas.tsx  # Thumbnail canvas for homepage cards
│       │   ├── GameUploadForm.tsx     # Upload form (game.py + metadata.json + metadata fields)
│       │   ├── VideoRecorder.tsx      # Record prompt, controls, preview modal, upload
│       │   ├── ConfirmModal.tsx       # Reusable confirm/cancel dialog
│       │   └── arcade/
│       │       └── ArcadeComponents.tsx  # Shared arcade components (StatPill, DPadButton, ArcadeCabinet, etc.)
│       ├── pages/
│       │   ├── HomePage.tsx           # Public: arcade-themed game catalog + "Play Your Own" tab
│       │   ├── PublicPlayPage.tsx      # Public: /play/:gameId — arcade cabinet game player
│       │   ├── DirectPlayPage.tsx     # Public: /play-direct — ephemeral upload-and-play (arcade themed)
│       │   ├── LoginPage.tsx          # Shared login for all roles → redirects to /dashboard
│       │   ├── RoleDashboardPage.tsx  # Role-specific dashboard home (Tasker/QL/PL/SuperAdmin views)
│       │   ├── MyGamesPage.tsx        # Tasker: game list, submit for review, update files, bulk download
│       │   ├── ReviewQueuePage.tsx    # QL/PL: review queue, approve/reject (2-step confirm), bulk download
│       │   ├── TeamManagementPage.tsx # Team management: create users, assign leads, multi-lead picker
│       │   ├── TeamMemberDetailPage.tsx  # QL/PL detail: games, gameplay stats, tasker list
│       │   ├── TaskerDetailPage.tsx   # Tasker detail: game history timeline, gameplay stats, files
│       │   ├── ProfilePage.tsx        # User profile + password change
│       │   ├── SettingsPage.tsx       # Super admin: recording toggle, theme
│       │   ├── DashboardPage.tsx      # Super admin: analytics stats, charts, recent games
│       │   ├── GamesPage.tsx          # Super admin: all games table, toggle, delete, sync
│       │   ├── GameDetailPage.tsx     # Super admin: game detail — overview, analytics, source, sessions, videos
│       │   ├── GameUploadPage.tsx     # Upload new game
│       │   ├── GamePlayPage.tsx       # Play game (authenticated)
│       │   ├── RequestedGamesPage.tsx # Super admin: game request management
│       │   ├── TempGamesPage.tsx      # Super admin: ephemeral session log
│       │   └── UsersPage.tsx          # Super admin: user management — create, edit, delete
│       └── types/
│           └── index.ts               # Client-side type re-exports
│
├── server/                       # Express backend
│   ├── index.ts                  # Entry point: middleware, routes, Vite/static setup, listen
│   ├── db.ts                     # PostgreSQL pool + initDB (10 tables) + migration + helpers
│   ├── routes.ts                 # Route registration: health check + 9 routers
│   ├── vite.ts                   # Vite dev middleware integration + SPA fallback
│   ├── middleware/
│   │   ├── auth.ts               # JWT, bcrypt, requireRole(), requireAdmin, ensureDefaultAdmin
│   │   ├── asyncHandler.ts       # Wraps async route handlers for error propagation
│   │   └── errorHandler.ts       # AppError class + Express error handler
│   ├── routes/
│   │   ├── auth.ts               # POST /login, GET /me, POST /register (role-aware)
│   │   ├── games.ts              # 7 public + 13 admin endpoints (CRUD, files, toggle, source, sync, videos, bulk-download)
│   │   ├── player.ts             # 3 public + 3 ephemeral + 4 admin + 1 palette endpoint
│   │   ├── analytics.ts          # 4 auth + 5 admin endpoints (dashboard, game stats, sessions, export)
│   │   ├── users.ts              # 4 admin + 1 self-password-change endpoint
│   │   ├── requests.ts           # 1 public + 5 admin endpoints (submit, list, review, source, files, delete)
│   │   ├── approval.ts           # Approval workflow: submit, ql-review, pl-review, admin-approve, audit log
│   │   ├── teams.ts              # Team management: my-team, assign, unassigned, member detail, tasker detail
│   │   └── notifications.ts      # Notification CRUD + SSE stream endpoint
│   ├── services/
│   │   ├── GamePythonBridge.ts   # Python subprocess manager (spawn, NDJSON, timeout, kill)
│   │   ├── NotificationService.ts # Create/query notifications + audit log entries + SSE emit
│   │   └── teamHelpers.ts        # Junction table helpers: getUserLeadIds, getUserLeadUsernames
│   └── python/
│       └── game_runner.py        # Python NDJSON bridge: init, action, reset, quit
│
├── shared/
│   └── types.ts                  # Shared TypeScript types (User, Game, UserRole, ApprovalStatus, Notification, AuditLogEntry, etc.)
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

## User Roles & Team Hierarchy

The platform supports 4 user roles with a hierarchical team structure:

| Role | Description | Can Create |
|------|-------------|------------|
| **Super Admin** | Full platform access. Manages all users, games, settings. | Any role |
| **PL** (Project Lead) | Final approval authority. Manages QLs and their taskers. | Tasker, QL |
| **QL** (Quality Lead) | First-level reviewer. Manages a team of taskers. | Tasker |
| **Tasker** | Uploads and submits games for review. | — |

### Team Assignment

- Taskers report to one or more QLs
- QLs report to one or more PLs
- Assignments are many-to-many via the `user_team_leads` junction table
- Super Admin can assign anyone; PLs can assign taskers to their QLs and QLs to themselves; QLs can add unassigned taskers to their team
- Username format: `prefix@ethara.ai`

---

## System Flows

### Flow 1: Public User Plays a Game

1. User opens the homepage (`/`), which fetches `GET /api/games/public` to display active games as an arcade-themed grid of preview cards.
2. User clicks a game card. A name modal appears prompting for an optional player name.
3. User enters name and clicks Play. Browser navigates to `/play/:gameId?name=<name>`.
4. `PublicPlayPage` mounts inside an arcade cabinet UI and calls `POST /api/player/public/start` with `{game_id, player_name}`.
5. Server looks up the game in the DB, spawns a `GamePythonBridge` subprocess, calls `bridge.init()` which loads the game `.py` file via `importlib`, instantiates the game class, and performs an initial `RESET` action.
6. A `play_sessions` row is inserted. The initial frame (grid, width, height, state, level, available_actions) is returned to the client.
7. A `SessionTracker` is initialized server-side to track per-level timing, actions, game_overs, and resets.
8. The `GameCanvas` component renders the grid inside the arcade cabinet screen area. The player uses keyboard (WASD/Arrow keys/Space) or clicks grid cells to send actions.
9. Each action calls `POST /api/player/public/action/:sessionGuid` with `{action, x?, y?}`. The server forwards the action to the Python bridge, receives the updated frame, detects level transitions and game_over events, updates `SessionTracker` counters, persists to `play_sessions`, and returns the frame.
10. On level clear (level number increases), the completed level's stats (actions, time, game_overs, resets) are recorded in `completedLevels`. The client shows a glitch-effect level clear animation.
11. On WIN or GAME_OVER terminal state, the session's `ended_at` is set, `games.total_plays` is incremented (and `total_wins` on WIN). The client shows a game over/win overlay with final stats.
12. When the user leaves the page, `POST /api/player/public/end/:sessionGuid` is called, which kills the Python bridge, cleans up in-memory state, and marks unfinished sessions as GAME_OVER.

### Flow 2: Game Upload & Approval Workflow

1. Tasker logs in at `/login`, redirected to `/dashboard`.
2. Navigates to Upload, submits `game.py` + `metadata.json`. Server creates game with `approval_status = 'draft'`, `is_active = false`.
3. Tasker goes to My Games, sees the draft game, clicks "Submit for Review". Optionally adds a note for the reviewer.
4. Server sets `approval_status = 'pending_ql'`, auto-assigns the tasker's QL(s), creates an audit entry, and sends real-time notification to the assigned QL.
5. QL sees the game in their Review Queue with "ACTION NEEDED" badge. Can view description, game rules, test play the game, and download files.
6. QL clicks Approve (2-step confirmation modal with optional remarks) → `approval_status = 'pending_pl'`, PL assigned from QL's lead chain. Or Reject with required reason → `approval_status = 'rejected'`, tasker notified.
7. PL sees QL-approved games in their Review Queue. Can test play, download, review.
8. PL clicks Approve → `approval_status = 'approved'`, `is_active = true`. Game now appears on public homepage. Or Reject → tasker and QL both notified.
9. Super Admin can bypass the workflow via "Admin Approve" at any stage.
10. Rejected games can be resubmitted by the tasker after fixing issues.

### Flow 3: File Update with Approval Reset

When files are updated on an already-submitted game, the approval resets based on who updates:

| Updater | New Status | Game Active | Notified |
|---------|-----------|-------------|----------|
| Tasker | `pending_ql` | No | QL |
| QL | `pending_pl` | No | PL + Tasker |
| PL | stays `approved` | Yes | Tasker + QL |
| Super Admin | no change | no change | — |

### Flow 4: Direct Play (Ephemeral)

1. User opens the homepage and clicks the "Play Your Own" tab, or navigates to `/play-direct`.
2. Uploads `game.py` + `metadata.json` and optionally enters a player name.
3. `POST /api/player/ephemeral/start` creates a temp directory in the OS temp folder, writes the uploaded files, spawns a `GamePythonBridge`, and inserts a row into `temp_game_sessions` with a 24-hour `expires_at`.
4. Session GUID is prefixed with `eph_` to distinguish ephemeral sessions.
5. Actions go through `POST /api/player/ephemeral/action/:sessionGuid`, updating `temp_game_sessions` including an `action_log` array.
6. On end (`POST /api/player/ephemeral/end/:sessionGuid`), the Python bridge is killed, the temp directory is deleted, and the session is marked as GAME_OVER.
7. No permanent `games` row is created. Sessions remain in `temp_game_sessions` until an admin deletes them.

### Flow 5: Real-Time Notifications

1. On login, `NotificationBell` component opens an SSE connection to `GET /api/notifications/stream?token=<jwt>`.
2. Server holds the connection open with 30-second heartbeats.
3. When any approval action, team assignment, or game update occurs, `NotificationService.createNotification()` inserts a DB record and calls `emitToUser()` which pushes to the SSE stream.
4. Client receives the event, increments the unread badge count, and plays a Web Audio API chime (dual-oscillator 880→1046Hz + 660→784Hz, 300ms).
5. Sound can be toggled on/off (persisted to localStorage).
6. Falls back to 15-second polling if SSE connection fails.

### Flow 6: Video Recording

Video recording streams directly to the server in real-time. **Zero client RAM/storage is used** — each 2-second chunk is uploaded and discarded immediately. This ensures recording works on low-end devices without crashes.

**Architecture:**
```
Browser                              Server
  │                                    │
  │ 1. Game starts → Record prompt     │
  │                                    │
  │ 2. User clicks "Record"            │
  │    getDisplayMedia() → tab capture  │
  │                                    │
  │ 3. POST /video/start ─────────────►│ Creates empty file on disk
  │    ◄──── { recording_id }          │
  │                                    │
  │ 4. Every 2 seconds:                │
  │    MediaRecorder.ondataavailable   │
  │    POST /video/chunk ─────────────►│ appendFileSync(chunk)
  │    (chunk discarded from RAM)      │ (writes directly to disk)
  │                                    │
  │ 5. Game ends (WIN/GAME_OVER)       │
  │    Auto-stop after 1.5s delay      │
  │    POST /video/end ───────────────►│ Finalizes file
  │    ◄──── { filename, size }        │
  │                                    │
  │ 6. "Saved" indicator shown         │
  │    Download link points to server  │
  │    Admin can view in Videos tab    │
```

**Quality settings:**

| Setting | Value | Reason |
|---------|-------|--------|
| Resolution | 1280x720 (720p) | Excellent for grid games, low resource usage |
| Frame rate | 30 fps | Smooth enough, half the data of 60fps |
| Bitrate | 4 Mbps | High quality for flat-color grid content |
| Chunk interval | 2 seconds | Efficient upload batching |
| Max duration | 5 minutes | Auto-stops to prevent runaway recordings |
| File size | ~30 MB/min | Half of 1080p/60fps |
| Client RAM | < 5 MB constant | Chunks are uploaded and discarded immediately |

**Codec priority:** MP4 H.264 (Chrome 120+, Edge, Safari) → WebM VP9 → WebM VP8

**Recording is only available for live games** (not ephemeral "Play Your Own"). The prompt shows "Not available" for ephemeral games since there's no server-side game directory.

### Flow 7: Admin Game Management

1. **Upload**: `/dashboard/upload` — upload `game.py` + `metadata.json` with optional metadata fields. Server validates, writes to `environment_files/`, creates DB record. Super admin uploads are auto-approved and active; other roles start as `draft`.
2. **Edit metadata**: `PUT /api/games/:gameId` — update name, description, rules, owner, links, fps, tags, active status.
3. **Replace files**: `PUT /api/games/:gameId/files` — upload new `game.py` and/or `metadata.json`. Triggers role-based approval reset (see Flow 3).
4. **Toggle active**: `PATCH /api/games/:gameId/toggle` — show/hide from public homepage.
5. **View source**: `GET /api/games/:gameId/source` — returns game Python source and parsed metadata JSON.
6. **Sync from filesystem**: `POST /api/games/sync-local` — scans `environment_files/` for directories with `metadata.json` not yet in the DB and creates game records.
7. **View sessions**: Analytics tab shows per-game stats; Sessions tab shows per-player cards with level-by-level breakdown.
8. **Export**: `GET /api/analytics/export/:gameId` — downloads `.xlsx` with session data + summary sheet. Supports date filters.
9. **Manage recordings**: List, stream, and delete `.webm` recordings.
10. **Bulk download**: `POST /api/games/bulk-download` — download multiple games as a combined ZIP with per-game folders.

### Flow 8: Reset Behavior

The RESET action (keyboard `R`) has context-dependent behavior tracked by `SessionTracker.actionsSinceReset`:

| Condition | Press R | Result |
|-----------|---------|--------|
| Player has made moves on current level (`actionsSinceReset > 0`) | 1st R | Reset same level. Level timer restarts. `levelResets++`. `actionsSinceReset` resets to 0. |
| Player has NOT moved since last reset (`actionsSinceReset === 0`) | 2nd R (double reset) | Full restart to Level 1. `completedLevels` cleared. Session `startTime` resets. All per-level counters reset. |
| After GAME_OVER | RESET | Continues same level. `levelGameOvers` was already incremented on the GAME_OVER transition. |

---

## Arcade UI Theme

Public-facing pages use a retro CRT arcade cabinet visual theme:

- **Pixel font**: "Press Start 2P" (Google Fonts) for branding, headers, and labels. Readable sans-serif for body text.
- **Color palette**: Neon colors (cyan, magenta, green, yellow, orange, pink, blue) for accents. Cabinet colors (body, panel, bezel, trim, screen) for structural elements.
- **CRT effects**: Scanline overlay, vignette, screen bezel, and subtle flicker animations via `arcade.css`.
- **Game player**: Canvas framed inside an arcade cabinet shape with marquee header, side art panels on large screens, control panel with D-pad and action buttons, coin slot decorations.
- **Homepage**: Dark background with starfield, neon-accented game cards, pixel font for section headers, clean readable text for game names and descriptions.
- **Dashboard pages**: Clean modern dark theme (not arcade-styled) for usability.

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

All tables are created automatically on server start via `initDB()` in `server/db.ts`. Existing databases are migrated with `migrateExistingData()`.

### `users`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `username` | `VARCHAR(50)` | UNIQUE NOT NULL | Login name (format: `prefix@ethara.ai`) |
| `hashed_password` | `VARCHAR(255)` | NOT NULL | bcrypt hash |
| `email` | `VARCHAR(255)` | UNIQUE | Optional email |
| `display_name` | `VARCHAR(100)` | | Human-readable name |
| `role` | `VARCHAR(20)` | DEFAULT 'tasker' | `tasker`, `ql`, `pl`, or `super_admin` |
| `is_admin` | `BOOLEAN` | DEFAULT FALSE | Legacy admin flag (super_admin = true) |
| `is_active` | `BOOLEAN` | DEFAULT TRUE | Account active flag |
| `team_lead_id` | `TEXT` | FK → users(id) | Primary lead (legacy, see `user_team_leads` for many-to-many) |
| `allowed_pages` | `JSONB` | DEFAULT '[]' | Legacy page-level access control list |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |
| `updated_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |

### `user_team_leads`

Many-to-many junction table for team assignments.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `user_id` | `TEXT` | FK → users(id) ON DELETE CASCADE | The team member |
| `lead_id` | `TEXT` | FK → users(id) ON DELETE CASCADE | The lead they report to |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |

UNIQUE constraint on `(user_id, lead_id)`. Indexes on `user_id` and `lead_id`.

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
| `approval_status` | `VARCHAR(20)` | DEFAULT 'draft' | `draft`, `pending_ql`, `ql_approved`, `pending_pl`, `approved`, `rejected` |
| `assigned_ql_id` | `TEXT` | FK → users(id) | Assigned Quality Lead |
| `assigned_pl_id` | `TEXT` | FK → users(id) | Assigned Project Lead |
| `rejection_reason` | `TEXT` | | Reason for rejection |
| `rejection_by` | `TEXT` | FK → users(id) | Who rejected |
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
| `uploaded_by` | `TEXT` | FK → users(id) | User who uploaded |

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

### `notifications`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `user_id` | `TEXT` | FK → users(id) ON DELETE CASCADE | Recipient |
| `type` | `VARCHAR(50)` | NOT NULL | `game_submitted`, `game_approved_ql`, `game_approved_pl`, `game_rejected`, `game_resubmitted`, `team_assigned`, `game_updated` |
| `title` | `VARCHAR(200)` | NOT NULL | Notification title |
| `message` | `TEXT` | | Notification body |
| `game_id` | `TEXT` | FK → games(id) ON DELETE SET NULL | Related game (optional) |
| `is_read` | `BOOLEAN` | DEFAULT FALSE | Read status |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |

Indexes on `(user_id, is_read)` and `(created_at DESC)`.

### `audit_log`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `TEXT` | PK | UUID |
| `game_id` | `TEXT` | FK → games(id) ON DELETE CASCADE | Related game |
| `user_id` | `TEXT` | FK → users(id) ON DELETE SET NULL | Actor |
| `action` | `VARCHAR(50)` | NOT NULL | `game_uploaded`, `game_updated`, `game_submitted_for_review`, `game_approved_by_ql`, `game_rejected_by_ql`, `game_approved_by_pl`, `game_rejected_by_pl`, `game_approved_by_admin`, `game_resubmitted`, `game_version_updated` |
| `details` | `JSONB` | | Additional context (reason, message, previous_status, etc.) |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | |

Indexes on `(game_id, created_at DESC)` and `(user_id)`.

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
CREATE INDEX idx_notifications_user ON notifications(user_id, is_read);
CREATE INDEX idx_notifications_date ON notifications(created_at DESC);
CREATE INDEX idx_audit_game       ON audit_log(game_id, created_at DESC);
CREATE INDEX idx_audit_user       ON audit_log(user_id);
CREATE INDEX idx_team_leads_user  ON user_team_leads(user_id);
CREATE INDEX idx_team_leads_lead  ON user_team_leads(lead_id);
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
| `POST` | `/auth/login` | None | Login with `{username, password}`, returns JWT + user with role |
| `GET` | `/auth/me` | JWT | Get current user profile (includes `role`, `display_name`, `team_lead_ids`) |
| `POST` | `/auth/register` | JWT (Super Admin / PL / QL) | Create new user. Accepts `username`, `password`, `email`, `display_name`, `role`, `team_lead_ids[]`. QLs can only create taskers; PLs can create taskers and QLs. |

### Approval — `/api/approval`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/approval/my-games` | JWT (Tasker) | List own games with approval details |
| `GET` | `/approval/ql-queue` | JWT (QL) | Games assigned to this QL for review |
| `GET` | `/approval/pl-queue` | JWT (PL) | Games assigned to this PL for final review |
| `GET` | `/approval/all` | JWT (Super Admin) | All games with approval details |
| `POST` | `/approval/:gameId/submit-for-review` | JWT (Tasker) | Submit draft/rejected game for QL review. Optional `{message}` body. |
| `POST` | `/approval/:gameId/ql-review` | JWT (QL) | `{action: "approve"\|"reject", reason?}` |
| `POST` | `/approval/:gameId/pl-review` | JWT (PL) | `{action: "approve"\|"reject", reason?}` |
| `POST` | `/approval/:gameId/admin-approve` | JWT (Super Admin) | Bypass approval. Optional `{message}`. |
| `GET` | `/approval/:gameId/audit` | JWT | Audit log for game (tasker: own only, QL: assigned, PL/admin: all) |

### Teams — `/api/teams`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/teams/my-team` | JWT (QL/PL) | QL: their taskers. PL: their QLs + nested taskers. |
| `GET` | `/teams/all-users` | JWT (Super Admin) | All users with team lead info |
| `PUT` | `/teams/assign` | JWT (Super Admin/PL/QL) | `{user_id, lead_id, action?}`. `action='remove'` to unassign. |
| `GET` | `/teams/unassigned` | JWT (Super Admin/PL/QL) | Unassigned users eligible for team assignment |
| `GET` | `/teams/:userId/detail` | JWT (PL/Super Admin) | QL/PL detail with games, gameplay stats, tasker list. Supports `?from=&to=` date filters. |
| `GET` | `/teams/:userId/tasker-detail` | JWT (QL/PL/Super Admin) | Tasker detail with games, audit timeline, gameplay stats, file availability. Supports `?from=&to=`. |

### Notifications — `/api/notifications`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/notifications/stream` | JWT (query param `token`) | SSE stream for real-time notifications |
| `GET` | `/notifications/` | JWT | List notifications. Optional `?unread=true&limit=N`. Returns `{notifications, unread_count}`. |
| `GET` | `/notifications/unread-count` | JWT | Just the unread count |
| `PUT` | `/notifications/:id/read` | JWT | Mark single notification as read |
| `PUT` | `/notifications/read-all` | JWT | Mark all as read |

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
| `POST` | `/games/public/:gameId/video/start` | None | Start streaming recording session |
| `POST` | `/games/public/:gameId/video/chunk` | None | Append 2s video chunk (multipart, `chunk` field) |
| `POST` | `/games/public/:gameId/video/end` | None | Finalize recording on server |
| `POST` | `/games/public/:gameId/video` | None | Legacy single-file upload (multipart, `video` field) |

**Authenticated (role-based):**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/games/` | JWT (Super Admin) | List all games (optional `?active_only=true`) |
| `GET` | `/games/:gameId` | JWT (all roles) | Get game details. Taskers: own games only. |
| `POST` | `/games/upload` | JWT (all roles) | Upload new game. Super admin: auto-approved. Others: draft. |
| `PUT` | `/games/:gameId` | JWT (Super Admin) | Update game metadata |
| `PUT` | `/games/:gameId/files` | JWT (all roles) | Replace game files. Triggers role-based approval reset. |
| `PATCH` | `/games/:gameId/toggle` | JWT (Super Admin) | Toggle `is_active` |
| `DELETE` | `/games/:gameId` | JWT (Super Admin) | Delete game + sessions + files |
| `DELETE` | `/games/:gameId/sessions` | JWT (Super Admin) | Clear sessions + reset counters |
| `GET` | `/games/:gameId/source` | JWT (all roles) | View `.py` source + parsed metadata |
| `GET` | `/games/:gameId/download` | JWT (all roles) | Download game files as ZIP |
| `POST` | `/games/bulk-download` | JWT (all roles) | Download multiple games as ZIP. Body: `{game_ids: string[]}` (max 100). |
| `POST` | `/games/sync-local` | JWT (Super Admin) | Scan filesystem, create DB records for new games |
| `GET` | `/games/:gameId/videos` | JWT (Super Admin) | List recordings |
| `GET` | `/games/:gameId/videos/:filename` | JWT (Super Admin) | Stream recording file |
| `DELETE` | `/games/:gameId/videos/:filename` | JWT (Super Admin) | Delete recording |

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
| `POST` | `/player/start` | JWT | Start authenticated session (allows playing inactive games for testing/review) |
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

**Super Admin:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/analytics/temp-sessions` | Super Admin | List all ephemeral sessions |
| `DELETE` | `/analytics/temp-sessions` | Super Admin | Delete all ephemeral sessions |
| `GET` | `/analytics/export/:gameId` | Super Admin | Export sessions as .xlsx (filters: `?filter=all\|today\|date\|range`) |
| `GET` | `/analytics/export-all` | Super Admin | Export all games' sessions as .xlsx |
| `GET` | `/analytics/export-games` | Super Admin | Export games list as .xlsx |

### Users — `/api/users`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/users/` | Super Admin | List all users (with role, display_name, team_lead_ids) |
| `GET` | `/users/:userId` | Super Admin | Get single user |
| `PUT` | `/users/:userId` | Super Admin | Update user (email, role, display_name, is_active, password, team_lead_ids) |
| `DELETE` | `/users/:userId` | Super Admin | Delete user (cannot delete self or protected user) |
| `PUT` | `/users/me/change-password` | JWT | Change own password: `{current_password, new_password}` |
| `POST` | `/users/protected/change-password` | JWT | Protected admin password change (requires `secret_code`) |

### Requests — `/api/requests`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/requests/submit` | None | Submit game for review (multipart + form fields) |
| `GET` | `/requests/` | Super Admin | List requests by status (`?status=pending\|approved\|rejected\|all`) |
| `GET` | `/requests/:requestId` | Super Admin | Get request details |
| `GET` | `/requests/:requestId/source` | Super Admin | View submitted source code + metadata |
| `GET` | `/requests/:requestId/files/:fileType` | Super Admin | Download submitted file (`game` or `metadata`) |
| `POST` | `/requests/:requestId/review` | Super Admin | Approve or reject: `{action: "approve"\|"reject", admin_note?}` |
| `DELETE` | `/requests/:requestId` | Super Admin | Delete request + cleanup files |

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

## Frontend Pages

| Route | Component | Auth | Description |
|-------|-----------|------|-------------|
| `/` | `HomePage` | None | Arcade-themed game catalog + "Play Your Own" tab |
| `/play/:gameId` | `PublicPlayPage` | None | Arcade cabinet game player with canvas, controls, timer, recorder |
| `/play-direct` | `DirectPlayPage` | None | Ephemeral upload-and-play (arcade themed) |
| `/login` | `LoginPage` | None | Shared login for all roles → redirects to `/dashboard` |
| `/dashboard` | `RoleDashboardPage` | JWT | Role-specific home (Tasker/QL/PL/SuperAdmin views with stats) |
| `/dashboard/my-games` | `MyGamesPage` | JWT | Tasker: game list, submit for review, update files, bulk download |
| `/dashboard/upload` | `GameUploadPage` | JWT | Upload new game (all roles) |
| `/dashboard/review` | `ReviewQueuePage` | JWT (QL/PL/SA) | Review queue: approve/reject (2-step confirm), game rules, bulk download |
| `/dashboard/team` | `TeamManagementPage` | JWT (QL/PL/SA) | Team management: create users, assign leads, multi-lead picker |
| `/dashboard/team/:userId` | `TeamMemberDetailPage` | JWT (PL/SA) | QL/PL detail: games by status, gameplay stats, tasker list |
| `/dashboard/team/tasker/:userId` | `TaskerDetailPage` | JWT (QL/PL/SA) | Tasker detail: game history timeline, gameplay stats, file download |
| `/dashboard/profile` | `ProfilePage` | JWT | User profile + change password |
| `/dashboard/games/:gameId/play` | `GamePlayPage` | JWT | Play game (authenticated, allows inactive games) |
| `/dashboard/settings` | `SettingsPage` | JWT (SA) | Recording toggle, theme |
| `/dashboard/games` | `GamesPage` | JWT (SA) | All games table: search, filter, toggle, delete, sync |
| `/dashboard/games/:gameId` | `GameDetailPage` | JWT (SA) | Game detail: overview, analytics, source, sessions, videos |
| `/dashboard/users` | `UsersPage` | JWT (SA) | User management: create, edit, delete |
| `/dashboard/requests` | `RequestedGamesPage` | JWT (SA) | Game request management |
| `/dashboard/temp-games` | `TempGamesPage` | JWT (SA) | Ephemeral session log |
| `/dashboard/admin-stats` | `DashboardPage` | JWT (SA) | Analytics dashboard with charts |
| `/dashboard/eval` | `EvalPage` | JWT (SA) | Eval runner |
| `/dashboard/logs` | `LogsPage` | JWT (SA) | System logs |
| `/dashboard/system` | `SystemPage` | JWT (SA) | System status |
| `/admin/*` | — | — | Redirects to `/dashboard` |

---

## Authentication & Authorization

### JWT Flow

1. Client sends `POST /api/auth/login` with `{username, password}`.
2. Server verifies bcrypt hash, returns `{access_token, token_type: "bearer"}` plus user object with `role`.
3. Client stores token in `localStorage` as `arc_token`.
4. All subsequent requests include `Authorization: Bearer <token>` via Axios interceptor.
5. On 401 response while on `/dashboard/*` or `/admin/*`, client clears token and redirects to `/login`.

### Role-Based Access

| Role | Dashboard | Upload | My Games | Review Queue | Team | Admin Pages | Settings |
|------|-----------|--------|----------|-------------|------|-------------|----------|
| **Super Admin** | Full analytics | Yes (auto-approved) | — | All games | All users | Full access | Yes |
| **PL** | PL overview | Yes | — | PL queue | QLs + taskers | — | — |
| **QL** | QL overview | Yes | — | QL queue | Own taskers | — | — |
| **Tasker** | Tasker overview | Yes | Yes | — | — | — | — |

### Middleware

- `authenticateToken` — Validates JWT, attaches `req.user`
- `requireAdmin` — Checks `is_admin` or `role === 'super_admin'`
- `requireRole(...roles: UserRole[])` — Factory that checks user's role against allowed list. `super_admin` always passes.

### Protected Super Admin

If `PROTECTED_USERNAME` is set in `.env`, a super admin account is created on first start. This account:
- Cannot be modified or deleted by other admins via the Users API.
- Password can only be changed via `POST /api/users/protected/change-password` with the correct `PROTECTED_SECRET_CODE`.

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

- Recording is only available for **live games** (not ephemeral "Play Your Own").
- The `recordings/` subdirectory is created automatically under the game's `local_dir`.
- Ensure the Node process has write permissions to `environment_files/`.
- Chunks are streamed every 2 seconds — if the browser tab closes mid-recording, any already-uploaded chunks will be on disk as a partial file.
- The 5-minute max duration auto-stops the recording to prevent runaway storage usage.
- If a recording file is empty (0 bytes), it is automatically deleted on session end.

### Recording causes lag or crash

This should not happen with the streaming architecture. If it does:
- The `getDisplayMedia()` API runs on the browser's media thread — zero main thread cost.
- Video chunks are POST'd every 2 seconds and immediately discarded from client RAM.
- Total client RAM usage stays under 5 MB regardless of recording length.
- Check browser DevTools Memory tab to confirm. If RAM grows, the issue is elsewhere (e.g., game engine, not recording).
- Video uploads are limited by Express body size (`50mb` limit configured in `server/index.ts`).
