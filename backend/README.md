# ARC-AGI-3 Internal - Backend

FastAPI backend for managing ARC-AGI-3 game environments, player sessions, analytics, game request approvals, and Excel exports.

---

## Quick Start

```bash
cd backend
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload
```

On first startup the server will:
1. Create all database tables via SQLAlchemy
2. Create the default admin user
3. Create the protected super admin user
4. Sync local game metadata from disk to DB
5. Clean up expired ephemeral sessions (>24h)

**Health check:** http://localhost:8008/api/health
**Swagger docs:** http://localhost:8008/docs

---

## Environment Variables (`.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DB_HOST` | Yes | - | PostgreSQL host |
| `DB_PORT` | No | `5432` | PostgreSQL port |
| `DB_NAME` | Yes | - | Database name |
| `DB_USER` | Yes | - | Database user |
| `DB_PASSWORD` | Yes | - | Database password |
| `SECRET_KEY` | Yes | - | JWT signing secret (long random string) |
| `DEFAULT_ADMIN_USERNAME` | No | `admin` | Default admin username |
| `DEFAULT_ADMIN_PASSWORD` | Yes | - | Default admin password |
| `ENVIRONMENT_FILES_DIR` | No | `./environment_files` | Game files storage path |
| `CORS_ORIGINS` | No | `http://localhost:5173` | Comma-separated allowed origins |
| `ALGORITHM` | No | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `1440` | Token expiry in minutes (default 24h) |
| `PROTECTED_USERNAME` | Yes | - | Protected super admin username |
| `PROTECTED_SECRET_CODE` | Yes | - | Secret code for protected admin password reset |

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, middleware, router registration
│   ├── config.py                # Pydantic Settings (reads .env)
│   ├── database.py              # SQLAlchemy engine + session factory (PostgreSQL)
│   ├── auth.py                  # JWT auth, bcrypt, get_current_user/get_admin_user deps
│   ├── models.py                # DB models: User, Game, PlaySession, GameRequest, TempGameSession, GameAnalytics
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── routers/
│   │   ├── auth.py              # Login, register, /me
│   │   ├── games.py             # Game CRUD, upload, toggle, public listing, previews
│   │   ├── player.py            # Play sessions: public, ephemeral, admin
│   │   ├── analytics.py         # Dashboard, per-game stats, Excel export
│   │   ├── users.py             # User CRUD, protected password change
│   │   └── requests.py          # Game request submission + admin review
│   └── services/
│       ├── game_engine.py       # ARC engine wrapper: load game, step, reset, level skip
│       ├── game_manager.py      # File validation, upload to disk, directory scanning
│       └── analytics.py         # Stats aggregation, session queries, dashboard data
├── environment_files/           # Game files stored on disk
│   ├── <game_code>/<version>/   # e.g. cw45/v1/cw45.py + metadata.json
│   └── _requests/               # Pending game request files
├── requirements.txt
├── Dockerfile
└── .env
```

---

## Database Models

### User (`users`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Auto-generated |
| `username` | String(50) | Unique, indexed |
| `hashed_password` | String(255) | bcrypt hash |
| `email` | String(255) | Optional, unique |
| `is_admin` | Boolean | Admin flag (default false) |
| `is_active` | Boolean | Account active (default true) |
| `allowed_pages` | JSON | Page access list: `["dashboard","games","upload","requests","users"]` |
| `created_at` | DateTime | IST timestamp |

### Game (`games`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Auto-generated |
| `game_id` | String(50) | **Unique** game identifier (e.g. `cw45`, `ls20-v1`) |
| `name` | String(100) | Display name (defaults to game_id) |
| `description` | Text | Game description |
| `game_rules` | Text | How to play / rules text |
| `game_owner_name` | String(100) | Creator's name |
| `game_drive_link` | String(500) | Google Drive / download URL |
| `game_video_link` | String(500) | Demo video URL |
| `game_code` | String(10) | Short code extracted from game_id |
| `version` | String(20) | Default `v1` |
| `is_active` | Boolean | Public visibility (new uploads start as inactive) |
| `default_fps` | Integer | Frames per second (from metadata.json) |
| `tags` | JSON | Tag list |
| `baseline_actions` | JSON | Avg actions per level |
| `total_plays` | Integer | Aggregate play count |
| `total_wins` | Integer | Aggregate win count |
| `game_file_path` | String(500) | Absolute path to `.py` on disk |
| `metadata_file_path` | String(500) | Absolute path to `metadata.json` on disk |
| `local_dir` | String(500) | Directory containing both files |
| `uploaded_by` | FK -> User | Who uploaded it |

### PlaySession (`play_sessions`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Auto-generated |
| `game_id` | FK -> Game | Game being played |
| `user_id` | FK -> User | Null for anonymous/public players |
| `session_guid` | String(100) | Unique public session ID |
| `player_name` | String(100) | Display name for anonymous players |
| `state` | String(20) | `NOT_FINISHED`, `WIN`, `GAME_OVER` |
| `total_actions` | Integer | Total actions taken |
| `total_time` | Float | Total elapsed seconds (server-side) |
| `current_level` | Integer | Last level reached |
| `game_overs` | Integer | Number of deaths/game-overs |
| `level_stats` | JSON | `[{level, actions, time, completed, lives_used, game_overs, resets}]` |
| `action_log` | JSON | `[{action, timestamp, level}]` |
| `started_at` / `ended_at` | DateTime | Session timestamps |

### GameRequest (`game_requests`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID (PK) | Auto-generated |
| `game_id` | String(50) | **Unique** proposed game_id |
| `requester_name` | String(100) | Who submitted it |
| `requester_email` | String(255) | Contact email (optional) |
| `message` | Text | Note to admin |
| `description` | Text | Game description |
| `game_rules` | Text | How to play |
| `status` | String(20) | `pending`, `approved`, `rejected` |
| `admin_note` | Text | Admin review note |
| `game_file_content` | LargeBinary | `.py` file stored in DB |
| `metadata_file_content` | LargeBinary | `metadata.json` stored in DB |
| `game_file_path` | String(500) | Also stored on disk in `_requests/` |

### TempGameSession (`temp_game_sessions`)

Ephemeral "Play Your Own" sessions. Auto-deleted after 24 hours on startup.

| Column | Type | Description |
|--------|------|-------------|
| `session_guid` | String(50) | Unique public GUID |
| `game_id` | String(50) | Temp game identifier |
| `state` | String(20) | Game state |
| `total_time` | Float | Elapsed seconds |
| `level_stats` | JSON | Per-level breakdown |
| `action_log` | JSON | All actions taken |
| `expires_at` | DateTime | start + 24h, auto-cleanup |

---

## API Reference

### Auth `/api/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/login` | None | Login with username/password, returns JWT |
| GET | `/me` | User | Get authenticated user's profile |
| POST | `/register` | Admin | Create a new user account |

### Games `/api/games`

**Public (no auth):**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/public` | List all active games |
| GET | `/public/stats` | Platform stats (total plays, featured game, top 3) |
| GET | `/public/{game_id}` | Get a single active game |
| GET | `/public/{game_id}/plays` | Leaderboard / recent plays |
| GET | `/public/{game_id}/stats` | Per-game stats (win rate, top performer) |
| GET | `/public/{game_id}/preview` | Initial grid frame for thumbnails |

**Admin (auth required):**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List all games (including inactive) |
| GET | `/{game_id}` | Get game details |
| POST | `/upload` | Upload new game (multipart: game_file + metadata_file + form fields) |
| PUT | `/{game_id}` | Update metadata (name, description, rules, tags, etc.) |
| PUT | `/{game_id}/files` | Replace game.py and/or metadata.json on disk |
| PATCH | `/{game_id}/toggle` | Toggle active/inactive |
| DELETE | `/{game_id}` | Delete game + all files + all sessions |
| DELETE | `/{game_id}/sessions` | Clear sessions only (reset counters) |
| GET | `/{game_id}/source` | View game.py source code + metadata.json |
| POST | `/sync-local` | Scan environment_files/ dir and register new games |

### Player `/api/player`

**Public (anonymous play, data stored in DB):**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/public/start` | Start session. Body: `{game_id, seed, player_name?, start_level?}` |
| POST | `/public/action/{guid}` | Send action. Body: `{action, x?, y?}` |
| POST | `/public/end/{guid}` | End session |

**Ephemeral (upload + play, no game in DB, temp files):**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ephemeral/start` | Multipart: game_file + metadata_file. Returns initial frame. |
| POST | `/ephemeral/action/{guid}` | Send action |
| POST | `/ephemeral/end/{guid}` | End session + cleanup temp files |

**Admin (authenticated play):**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/start` | Start admin session |
| POST | `/action/{guid}` | Send action |
| GET | `/frame/{guid}` | Get current frame (no action) |
| POST | `/end/{guid}` | End session |
| GET | `/palette` | Get the 16-color ARC-AGI palette |

### Analytics `/api/analytics`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/dashboard` | User | Dashboard: totals, 7-day chart, game distribution |
| GET | `/game/{game_id}` | User | Per-game: plays, wins, avg time/actions, daily breakdown |
| GET | `/sessions` | User | Recent sessions (optional `game_id` filter) |
| GET | `/replay/{session_id}` | User | Full replay data (action_log, level_stats) |
| GET | `/temp-sessions` | Admin | List ephemeral session logs |
| DELETE | `/temp-sessions` | Admin | Delete all ephemeral logs |
| GET | `/export/{game_id}` | Admin | Export sessions as Excel (.xlsx) |
| GET | `/export-all` | Admin | Export all sessions as Excel |
| GET | `/export-games` | Admin | Export games list as Excel |

### Users `/api/users`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Admin | List all users |
| GET | `/{id}` | Admin | Get user |
| PUT | `/{id}` | Admin | Update user (role, status, pages, password) |
| DELETE | `/{id}` | Admin | Delete user |
| POST | `/protected/change-password` | Self | Protected admin password change (needs secret_code) |

### Game Requests `/api/requests`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/submit` | None | Public: submit game for review (multipart) |
| GET | `/` | Admin | List requests (filter: pending/approved/rejected/all) |
| GET | `/{id}` | Admin | Get request details |
| GET | `/{id}/source` | Admin | View submitted source code |
| GET | `/{id}/files/{type}` | Admin | Download submitted file (game or metadata) |
| POST | `/{id}/review` | Admin | Approve or reject. Body: `{action: "approve"|"reject", admin_note?}` |
| DELETE | `/{id}` | Admin | Delete request + files |

---

## Game Engine Details

The `GameEngineService` (`services/game_engine.py`) manages in-memory game instances.

### Loading Strategies (tried in order)

| # | Mode | What it does |
|---|------|-------------|
| 1 | `direct` | Dynamically imports the `.py` file via `importlib`, registers in `sys.modules` (fixes `@dataclass` bug in `arc_agi`), instantiates game class, uses `perform_action()` from arcengine |
| 2 | `arcade` | Uses `arc_agi.Arcade()` from the official SDK |
| 3 | `parsed` | Text-parses the `.py` for grid sizes and sprite pixel data (read-only, no execution) |
| 4 | `none` | Empty 8x8 grid fallback |

### Reset Logic
- **Press R once (has moved):** Resets current level only
- **Press R again (no moves since last reset):** Full restart to Level 1

### Per-level Stats Tracked
- `level` - level index
- `actions` - actions taken on this level
- `time` - seconds spent on this level
- `completed` - whether level was finished
- `lives_used` - deaths on this level
- `game_overs` - game over events
- `resets` - reset count

### Session Limits
- Max 600 concurrent sessions
- 30-minute idle timeout
- 500 max action log entries per session

---

## Game File Format

```
environment_files/<game_code>/<version>/
├── <game_code>.py        # Python class extending ARCBaseGame
└── metadata.json         # Game metadata
```

### metadata.json

```json
{
  "game_id": "cw45",
  "default_fps": 10,
  "baseline_actions": [15, 20, 25],
  "tags": ["puzzle", "navigation"]
}
```

**Required:** `game_id` must match `<letters><digits>` (e.g. `cw45`) or `<code>-v<n>` (e.g. `ls20-v1`).

**Allowed keys only:** `game_id`, `default_fps`, `baseline_actions`, `tags`, `local_dir`. Unknown keys are rejected.

### game.py

Must contain a class extending `ARCBaseGame` with a `step()` method:

```python
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

class Cw45(ARCBaseGame):
    def __init__(self, seed=0):
        camera = Camera(background=5, width=9, height=9)
        super().__init__(game_id="cw45", levels=levels, camera=camera)

    def step(self):
        # Handle actions, check win/lose
        self.complete_action()
```

**Validation rules:** No comments (`#`) or docstrings (`"""`) allowed in the game file.

---

## Unique Game ID Enforcement

The `game_id` is unique across the entire system:

1. **On game upload:** Checks `games` table - rejects if game_id already exists
2. **On request submit:** Checks both `games` table AND pending `game_requests` - rejects if either has the same game_id
3. **On request approve:** Checks `games` table again before creating the game

---

## Docker

```bash
docker build -t arc-backend .
docker run -p 8008:8008 --env-file .env arc-backend
```

| Property | Value |
|----------|-------|
| Base image | `python:3.12-slim` |
| Exposed port | `8008` |
| Workers | 1 |
| Concurrency limit | 600 |
| Created dirs | `/app/environment_files`, `/app/data` |

---

## ARC-AGI-3 Color Palette

Source: `app.py -> _PALETTE_HEX`

| Index | Name | Hex | RGB |
|-------|------|-----|-----|
| 0 | White | `#FFFFFF` | (255, 255, 255) |
| 1 | Off-white | `#CCCCCC` | (204, 204, 204) |
| 2 | Light Grey | `#999999` | (153, 153, 153) |
| 3 | Grey | `#666666` | (102, 102, 102) |
| 4 | Dark Grey | `#333333` | (51, 51, 51) |
| 5 | Black | `#000000` | (0, 0, 0) |
| 6 | Magenta | `#E53AA3` | (229, 58, 163) |
| 7 | Pink | `#FF7BCC` | (255, 123, 204) |
| 8 | Red | `#F93C31` | (249, 60, 49) |
| 9 | Blue | `#1E93FF` | (30, 147, 255) |
| 10 | Light Blue | `#88D8F1` | (136, 216, 241) |
| 11 | Yellow | `#FFDC00` | (255, 220, 0) |
| 12 | Orange | `#FF851B` | (255, 133, 27) |
| 13 | Maroon | `#921231` | (146, 18, 49) |
| 14 | Green | `#4FCC30` | (79, 204, 48) |
| 15 | Purple | `#A356D6` | (163, 86, 214) |

Special: `-1` = transparent, `-2` = special transparent (not rendered).

Typical usage: `5` (Black) = background, `11` (Yellow) = player, `3`/`4` (Grey) = walls, `8` (Red) = danger, `14` (Green) = success.
