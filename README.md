# ARC-AGI-3 Internal Platform

Full-stack platform for uploading, testing, playing, and managing ARC-AGI-3 grid-based puzzle games with real-time analytics, game request approvals, and Excel exports.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PUBLIC WEBSITE (/)                              │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  Games Tab   │  │ Play Your Own│  │   Request Upload Tab         │  │
│  │              │  │              │  │                              │  │
│  │ Browse games │  │ Upload .py + │  │ Submit game.py + metadata   │  │
│  │ Click to play│  │ metadata.json│  │ for admin review            │  │
│  │              │  │ Play instant │  │                              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────┘  │
│         │                 │                          │                   │
│         ▼                 ▼                          ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ /play/:id    │  │ /play-direct │  │ POST /api/requests/submit    │  │
│  │              │  │              │  │ (stored in _requests/ dir)   │  │
│  │ Canvas game  │  │ Ephemeral    │  │                              │  │
│  │ player with  │  │ game player  │  └──────────────────────────────┘  │
│  │ timer, level │  │ (temp files, │                                     │
│  │ stats, reset │  │  cleaned up) │                                     │
│  └──────────────┘  └──────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        ADMIN PANEL (/admin)                             │
│                                                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────────────┐  │
│  │  Dashboard   │ │   Games     │ │  Requests    │ │    Users       │  │
│  │             │ │   Table     │ │              │ │                │  │
│  │ Stats cards │ │             │ │ Pending list │ │ Create/edit    │  │
│  │ Charts      │ │ Search/     │ │ Approve →    │ │ Role mgmt     │  │
│  │ Recent      │ │ filter/     │ │   creates    │ │ Page access    │  │
│  │ Top played  │ │ toggle/     │ │   Game in DB │ │                │  │
│  │             │ │ delete/     │ │ Reject →     │ │                │  │
│  │             │ │ export      │ │   with note  │ │                │  │
│  └─────────────┘ └──────┬──────┘ └──────────────┘ └────────────────┘  │
│                          │                                              │
│                          ▼                                              │
│              ┌─────────────────────┐                                    │
│              │  Game Detail Page   │                                    │
│              │                     │                                    │
│              │ Tab 1: Overview     │  Info, stats, recent plays         │
│              │ Tab 2: Analytics    │  Plays, wins, avg time, daily      │
│              │ Tab 3: Source Code  │  View game.py + metadata.json      │
│              │ Tab 4: Sessions     │  Per-player with level breakdown   │
│              └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            BACKEND                                      │
│                                                                         │
│  FastAPI (port 8008)                                                    │
│  ├── /api/auth       → JWT login/register                               │
│  ├── /api/games      → Game CRUD + public listing + preview             │
│  ├── /api/player     → Public/ephemeral/admin play sessions             │
│  ├── /api/analytics  → Dashboard stats + per-game + Excel export        │
│  ├── /api/requests   → Game submission + admin review                   │
│  └── /api/users      → User management                                 │
│                                                                         │
│  Services:                                                              │
│  ├── GameEngineService  → Loads .py files, runs game logic              │
│  ├── GameManagerService → File validation + disk management             │
│  └── AnalyticsService   → Stats aggregation + queries                   │
│                                                                         │
│  Database: PostgreSQL                                                   │
│  Storage:  environment_files/ (game .py + metadata.json on disk)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Flow: How Everything Works

### Flow 1: Game Creator Submits a Game

```
Game Creator                 Website                    Backend                    Admin
     │                          │                          │                          │
     │  Opens homepage          │                          │                          │
     │  Clicks "Request Upload" │                          │                          │
     │  ─────────────────────►  │                          │                          │
     │                          │                          │                          │
     │  Fills form:             │                          │                          │
     │  - Name, email           │                          │                          │
     │  - Game owner name       │                          │                          │
     │  - Description, rules    │                          │                          │
     │  - game.py file          │                          │                          │
     │  - metadata.json file    │                          │                          │
     │  Clicks "Submit"         │                          │                          │
     │  ─────────────────────►  │                          │                          │
     │                          │  POST /api/requests/     │                          │
     │                          │  submit (multipart)      │                          │
     │                          │  ───────────────────────►│                          │
     │                          │                          │                          │
     │                          │                          │  Validates:              │
     │                          │                          │  - game_id unique?       │
     │                          │                          │  - metadata.json valid?  │
     │                          │                          │  - No pending duplicate? │
     │                          │                          │                          │
     │                          │                          │  Stores files:           │
     │                          │                          │  _requests/<game_id>/    │
     │                          │                          │  + binary in DB          │
     │                          │                          │                          │
     │                          │                          │  Creates GameRequest     │
     │                          │                          │  status = "pending"      │
     │                          │                          │                          │
     │  "Submitted for review!" │                          │                          │
     │  ◄─────────────────────  │                          │                          │
     │                          │                          │                          │
     │                          │                          │                          │
     │                          │                          │  Admin opens /admin/     │
     │                          │                          │  requests               │
     │                          │                          │  ◄─────────────────────  │
     │                          │                          │                          │
     │                          │                          │  Admin reviews:          │
     │                          │                          │  - View source code      │
     │                          │                          │  - Play test (ephemeral) │
     │                          │                          │  ◄─────────────────────  │
     │                          │                          │                          │
     │                          │                          │  Admin clicks "Approve"  │
     │                          │                          │  ◄─────────────────────  │
     │                          │                          │                          │
     │                          │                          │  Backend:                │
     │                          │                          │  1. Copies files to      │
     │                          │                          │     environment_files/   │
     │                          │                          │  2. Creates Game record  │
     │                          │                          │     (is_active=false)    │
     │                          │                          │  3. Deletes request      │
     │                          │                          │                          │
     │                          │                          │  Admin toggles active    │
     │                          │                          │  on Games page           │
     │                          │                          │  ◄─────────────────────  │
     │                          │                          │                          │
     │                          │  Game now visible        │                          │
     │                          │  on public homepage      │                          │
```

### Flow 2: Player Plays a Game

```
Player                       Website                    Backend
  │                             │                          │
  │  Opens homepage             │                          │
  │  Sees game grid             │                          │
  │  Clicks a game              │                          │
  │  ──────────────────────────►│                          │
  │                             │                          │
  │  Name modal appears         │                          │
  │  Types name (optional)      │                          │
  │  Clicks "Play"              │                          │
  │  ──────────────────────────►│                          │
  │                             │  Navigate to             │
  │                             │  /play/<game_id>?name=X  │
  │                             │                          │
  │                             │  POST /api/player/       │
  │                             │  public/start            │
  │                             │  {game_id, player_name}  │
  │                             │  ───────────────────────►│
  │                             │                          │
  │                             │                          │  Creates PlaySession
  │                             │                          │  Loads game engine:
  │                             │                          │    1. Import game.py
  │                             │                          │    2. Instantiate class
  │                             │                          │    3. perform_action(RESET)
  │                             │                          │    4. Extract grid frame
  │                             │                          │
  │                             │  Returns GameFrame:      │
  │                             │  {grid, width, height,   │
  │                             │   state, level, actions,  │
  │                             │   metadata: {timer,      │
  │                             │   level_stats, ...}}     │
  │  ◄──────────────────────────│  ◄───────────────────────│
  │                             │                          │
  │  Canvas renders grid        │                          │
  │  Timer starts               │                          │
  │                             │                          │
  │  Player presses WASD/       │                          │
  │  clicks grid/Space          │                          │
  │  ──────────────────────────►│                          │
  │                             │  POST /api/player/       │
  │                             │  public/action/<guid>    │
  │                             │  {action: "ACTION1"}     │
  │                             │  ───────────────────────►│
  │                             │                          │
  │                             │                          │  Calls game.perform_action()
  │                             │                          │  Extracts new grid
  │                             │                          │  Tracks level stats
  │                             │                          │  Checks win/game_over
  │                             │                          │
  │  Updated grid + stats       │                          │
  │  ◄──────────────────────────│  ◄───────────────────────│
  │                             │                          │
  │  ... continues playing ...  │                          │
  │                             │                          │
  │  Game ends (WIN or          │                          │
  │  GAME_OVER)                 │                          │
  │                             │                          │
  │  Timer stops                │                          │  Updates PlaySession:
  │  Game over overlay shows    │                          │  - state, total_time
  │  Final: time, actions,      │                          │  - level_stats
  │         level reached       │                          │  - Updates Game.total_plays
  │                             │                          │  - Updates Game.total_wins
```

### Flow 3: Admin Views Analytics

```
Admin                        Admin Panel                Backend
  │                             │                          │
  │  Opens /admin/games/<id>    │                          │
  │  ──────────────────────────►│                          │
  │                             │  GET /api/games/<id>     │
  │                             │  GET /api/analytics/     │
  │                             │      game/<id>           │
  │                             │  GET /api/analytics/     │
  │                             │      sessions?game_id=   │
  │                             │  ───────────────────────►│
  │                             │                          │
  │  Overview Tab:              │                          │
  │  - Total plays, wins       │                          │
  │  - Win rate, avg time       │                          │
  │  - Recent plays list        │                          │
  │                             │                          │
  │  Analytics Tab:             │                          │
  │  - 30-day stats             │                          │
  │  - Avg actions, avg time    │                          │
  │  - Daily breakdown table    │                          │
  │                             │                          │
  │  Sessions Tab:              │                          │
  │  - Per-player cards:        │                          │
  │    ┌────────────────────┐   │                          │
  │    │ Player: John       │   │                          │
  │    │ State: WIN         │   │                          │
  │    │ Time: 2:34         │   │                          │
  │    │ Actions: 47        │   │                          │
  │    │ Level reached: 4   │   │                          │
  │    │                    │   │                          │
  │    │ Level Breakdown:   │   │                          │
  │    │ Lv1: 12m 0:34 done│   │                          │
  │    │ Lv2: 15m 1:12 done│   │                          │
  │    │ Lv3:  8m 0:28 done│   │                          │
  │    │ Lv4: 12m 0:20     │   │                          │
  │    └────────────────────┘   │                          │
  │                             │                          │
  │  Clicks "Export Excel"      │                          │
  │  ──────────────────────────►│                          │
  │                             │  GET /api/analytics/     │
  │                             │  export/<id>?filter=all  │
  │                             │  ───────────────────────►│
  │                             │                          │
  │  Downloads .xlsx file       │                          │
  │  ◄──────────────────────────│  ◄───────────────────────│
```

---

## Quick Start

### Development

```bash
# Terminal 1: Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8008
- API Docs: http://localhost:8008/docs
- Default admin: credentials from `.env`

### Docker

```bash
docker-compose up --build
```

Services:
- **db** (PostgreSQL 16) - port 5432
- **backend** (FastAPI) - port 8008
- **frontend** (Nginx + React) - port 5173

---

## Game File Format

Each game consists of two files stored at `environment_files/<code>/<version>/`:

### metadata.json
```json
{
  "game_id": "cw45",
  "default_fps": 10,
  "baseline_actions": [15, 20, 25],
  "tags": ["puzzle", "navigation"]
}
```

### game.py
```python
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

sprites = { ... }
levels = [ Level(sprites=[...], grid_size=(9, 9)), ... ]

class Cw45(ARCBaseGame):
    def __init__(self, seed=0):
        camera = Camera(background=5, width=9, height=9)
        super().__init__(game_id="cw45", levels=levels, camera=camera)

    def step(self):
        if self.action.id == GameAction.ACTION1:  # Up
            ...
        self.complete_action()
```

---

## Color Palette (ARC-AGI-3)

| Index | Hex | Name | Typical Use |
|-------|-----|------|-------------|
| 0 | `#FFFFFF` | White | |
| 1 | `#CCCCCC` | Off-white | |
| 2 | `#999999` | Light Grey | |
| 3 | `#666666` | Grey | Walls |
| 4 | `#333333` | Dark Grey | Obstacles |
| 5 | `#000000` | Black | Background/floor |
| 6 | `#E53AA3` | Magenta | |
| 7 | `#FF7BCC` | Pink | |
| 8 | `#F93C31` | Red | Danger |
| 9 | `#1E93FF` | Blue | |
| 10 | `#88D8F1` | Light Blue | |
| 11 | `#FFDC00` | Yellow | Player |
| 12 | `#FF851B` | Orange | |
| 13 | `#921231` | Maroon | |
| 14 | `#4FCC30` | Green | Success |
| 15 | `#A356D6` | Purple | |

---

## Data Tracked Per Play Session

| Data | Where | Description |
|------|-------|-------------|
| `total_time` | PlaySession | Server-side elapsed seconds |
| `total_actions` | PlaySession | Total actions taken |
| `current_level` | PlaySession | Last level reached |
| `game_overs` | PlaySession | Death count |
| `level_stats` | PlaySession (JSON) | Per-level: time, actions, completed, deaths, resets |
| `action_log` | PlaySession (JSON) | Every action with timestamp |
| `player_name` | PlaySession | Display name (optional) |
| `state` | PlaySession | WIN / GAME_OVER / NOT_FINISHED |

---

## Reset Behavior

| Situation | Press R | Result |
|-----------|---------|--------|
| Player has moved on current level | 1st R | Reset **same level** |
| Player has NOT moved (just reset) | 2nd R | Full restart to **Level 1** |
| Game Over / Win | R | Starts fresh new game |

---

## Sharing This Document

**For the team:**
1. `README.md` (this file) - Overall architecture and flow
2. `backend/README.md` - Complete API reference, models, engine details
3. `frontend/README.md` - All routes, pages, components, controls
4. `http://localhost:8008/docs` - Interactive Swagger API docs

**Key things to know:**
- Game files are `.py` (class extending `ARCBaseGame`) + `metadata.json`
- `game_id` must be globally unique across games and pending requests
- New uploads start as **inactive** (admin must toggle active)
- Public players can play without any authentication
- Ephemeral "Play Your Own" sessions auto-delete after 24 hours
- All play data (time, actions, level stats) is tracked and exportable as Excel
