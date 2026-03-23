# ARC-AGI-3 Internal - Frontend

React SPA for the ARC-AGI-3 game platform. Provides a public game portal for playing games and an admin panel for management.

---

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

**Dev server:** http://localhost:5173
**API proxy:** All `/api` requests are proxied to `http://localhost:8008` (backend)

---

## Tech Stack

| Category | Package | Version |
|----------|---------|---------|
| Core | React | 18.3 |
| Routing | react-router-dom | 6.26 |
| HTTP | axios | 1.7 |
| Icons | lucide-react | 0.447 |
| Charts | recharts | 2.12 |
| Animation | framer-motion | 12.38 |
| CSS | Tailwind CSS | 3.4 |
| Build | Vite | 5.4 |

---

## Project Structure

```
frontend/src/
├── App.jsx                        # Route definitions (public + admin)
├── main.jsx                       # React root render
├── index.css                      # Tailwind imports + custom styles
├── api/
│   └── client.js                  # Axios instance + all API methods
├── hooks/
│   ├── useAuth.jsx                # Auth context (login, logout, user state)
│   └── useGameEngine.jsx          # Admin game session hook
├── components/
│   ├── Layout.jsx                 # Admin shell: collapsible sidebar + content area
│   ├── GameCanvas.jsx             # HTML5 Canvas grid renderer (16-color palette)
│   ├── GamePreviewCanvas.jsx      # Static game preview thumbnail (fetches initial frame)
│   ├── GameUploadForm.jsx         # Shared upload form (admin upload + public request)
│   └── ConfirmModal.jsx           # Reusable confirmation dialog (danger/warning/success)
├── pages/
│   ├── HomePage.jsx               # Public: game browser + direct play + request upload
│   ├── PublicPlayPage.jsx         # Public: full game player (registered games)
│   ├── DirectPlayPage.jsx         # Public: ephemeral game player (uploaded files)
│   ├── LoginPage.jsx              # Admin: login page
│   ├── DashboardPage.jsx          # Admin: analytics dashboard
│   ├── GamesPage.jsx              # Admin: games table + management
│   ├── GameDetailPage.jsx         # Admin: single game (info, analytics, source, sessions)
│   ├── GameUploadPage.jsx         # Admin: upload new game
│   ├── GamePlayPage.jsx           # Admin: play game (authenticated session)
│   ├── RequestedGamesPage.jsx     # Admin: review community game submissions
│   ├── TempGamesPage.jsx          # Admin: ephemeral session logs
│   └── UsersPage.jsx              # Admin: user management
```

---

## Routes

### Public Routes (no authentication)

| Path | Page | Description |
|------|------|-------------|
| `/` | `HomePage` | Game browser with 3 tabs: Games / Play Your Own / Request Upload |
| `/play/:gameId` | `PublicPlayPage` | Play a registered game (auto-starts, timer, level stats) |
| `/play-direct` | `DirectPlayPage` | Play an uploaded game instantly (ephemeral, nothing saved permanently) |

### Admin Routes (authentication required)

| Path | Page | Description |
|------|------|-------------|
| `/admin/login` | `LoginPage` | Admin login |
| `/admin` | `DashboardPage` | Analytics dashboard (stats, charts, recent games) |
| `/admin/games` | `GamesPage` | Game management table (search, filter, export, toggle, delete) |
| `/admin/games/upload` | `GameUploadPage` | Upload a new game |
| `/admin/games/:gameId` | `GameDetailPage` | Game detail (4 tabs: Overview, Analytics, Source, Sessions) |
| `/admin/games/:gameId/play` | `GamePlayPage` | Admin game player |
| `/admin/requests` | `RequestedGamesPage` | Review pending game submissions (approve/reject) |
| `/admin/temp-games` | `TempGamesPage` | Ephemeral "Play Your Own" session logs |
| `/admin/users` | `UsersPage` | User management (create, edit roles, page access, delete) |

### Access Control

- `ProtectedRoute` - redirects to `/admin/login` if not authenticated
- `PageGuard` - checks `user.allowed_pages` array. Admins bypass all guards.
- Available pages: `dashboard`, `games`, `upload`, `requests`, `users`

---

## API Client (`src/api/client.js`)

Axios instance with:
- Base URL: `/api`
- Auto-attaches JWT from `localStorage` key `arc_token`
- On 401 response: clears token, redirects to `/admin/login` (only on admin pages)

### API Method Groups

**`authAPI`** - Login, register, get current user

**`gamesAPI`** - Public game listing/stats/preview + Admin CRUD/upload/toggle/delete/export

**`playerAPI`** - Three player modes:
- `publicStart/publicAction/publicEnd` - anonymous play (stored in DB)
- `ephemeralStart/ephemeralAction/ephemeralEnd` - upload + play (temp files, no permanent storage)
- `start/action/frame/end` - admin authenticated play

**`analyticsAPI`** - Dashboard stats, per-game analytics, sessions, replay, Excel export

**`usersAPI`** - User CRUD, protected password change

**`requestsAPI`** - Public submit, admin list/review/approve/reject/delete

---

## Key Pages in Detail

### HomePage (`/`)

Three tabs:

| Tab | What it does |
|-----|-------------|
| **Games** | Searchable grid of active games with preview thumbnails, play counts, win rates. Click -> name modal -> `/play/:gameId`. Shows featured game hero, platform stats sidebar. |
| **Play Your Own** | Upload `game.py` + `metadata.json` -> play instantly. Nothing saved to DB permanently. Navigates to `/play-direct`. |
| **Request Upload** | Submit a game for admin review. Requires: name, game owner, description, rules, files. Goes to pending queue. |

### PublicPlayPage (`/play/:gameId`)

- Auto-starts on page load (no start screen)
- Canvas-based grid renderer with click support
- Live timer in header (auto-starts, auto-stops on win/game over)
- Per-level stats in sidebar (time, moves, deaths per level)
- Win overlay: trophy + sparkle particles + final stats
- Game Over overlay: skull + shake animation
- Keyboard controls: WASD/arrows, Space, R=reset, Ctrl+Z=undo, Click=ACTION6
- 50ms action throttle
- Auto-recovery if server session is lost

### Admin GamesPage (`/admin/games`)

Table view with columns:
- Game ID (with name/description)
- Status badge (Active/Inactive)
- Tags (max 2 shown, +N for more)
- Plays, Wins, FPS, Created date
- Action buttons: Play, Details, Toggle, Delete

Features: Search, filter tabs (All/Active/Inactive), Excel export (date filters), sync local.

### Admin GameDetailPage (`/admin/games/:gameId`)

4 tabs:
1. **Overview** - Game info (editable), file replacement, stats card, recent plays
2. **Analytics** - 30-day stats: plays, wins, win rate, avg time, avg actions, daily table
3. **Source Code** - View game.py and metadata.json
4. **Sessions** - Expandable session cards with per-level breakdown (time, moves, deaths, resets)

### RequestedGamesPage (`/admin/requests`)

- Filter: Pending / Approved / Rejected / All
- Expandable cards: requester info, game description, rules, links
- Actions: Play Test (navigates to DirectPlayPage), Approve, Reject (with note), View Source, Delete
- On approve: game is created as inactive, request is removed

---

## Game Controls (All Play Pages)

| Key | Action |
|-----|--------|
| W / Arrow Up | ACTION1 (Up) |
| S / Arrow Down | ACTION2 (Down) |
| A / Arrow Left | ACTION3 (Left) |
| D / Arrow Right | ACTION4 (Right) |
| Space / F | ACTION5 (Action) |
| Click on grid | ACTION6 (x,y coordinate) |
| Ctrl+Z / Cmd+Z | ACTION7 (Undo) |
| R | Reset (1st: same level, 2nd: level 1) |

All actions throttled at 50ms minimum interval.

---

## Color Palette (ARC-AGI-3)

Used in `GameCanvas.jsx` and `useGameEngine.jsx`:

| Index | Hex | Name |
|-------|-----|------|
| 0 | `#FFFFFF` | White |
| 1 | `#CCCCCC` | Off-white |
| 2 | `#999999` | Light Grey |
| 3 | `#666666` | Grey |
| 4 | `#333333` | Dark Grey |
| 5 | `#000000` | Black (background) |
| 6 | `#E53AA3` | Magenta |
| 7 | `#FF7BCC` | Pink |
| 8 | `#F93C31` | Red |
| 9 | `#1E93FF` | Blue |
| 10 | `#88D8F1` | Light Blue |
| 11 | `#FFDC00` | Yellow (player) |
| 12 | `#FF851B` | Orange |
| 13 | `#921231` | Maroon |
| 14 | `#4FCC30` | Green (success) |
| 15 | `#A356D6` | Purple |

---

## Admin Sidebar Navigation

| Section | Label | Path | Icon |
|---------|-------|------|------|
| Overview | Dashboard | `/admin` | LayoutDashboard |
| Games | All Games | `/admin/games` | Gamepad2 |
| Games | Temp Games | `/admin/temp-games` | Timer |
| Games | Upload | `/admin/games/upload` | Upload |
| Games | Requests | `/admin/requests` | Inbox |
| Team | Users | `/admin/users` | Users |

Sidebar is collapsible (240px / 68px). Nav items filtered by `user.allowed_pages`.

---

## Docker

```bash
docker build -t arc-frontend .
docker run -p 80:80 arc-frontend
```

Two-stage build:
1. **Build** (`node:20-alpine`): `npm install` + `npm run build`
2. **Serve** (`nginx:alpine`): serves `/dist` via nginx with SPA fallback + API proxy

Note: needs an `nginx.conf` for:
- `try_files $uri /index.html` (SPA routing)
- `location /api/ { proxy_pass http://backend:8008; }` (API reverse proxy)

---

## Sharing This Document

To share with your team:
1. This README covers the full frontend architecture
2. See `../backend/README.md` for the backend API reference
3. See `../README.md` for the overall system architecture and flow diagrams
4. The Swagger docs at `http://localhost:8008/docs` provide interactive API testing
