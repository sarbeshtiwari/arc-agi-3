# Client — ARC-AGI-3 Internal Platform

## Overview

React 18 + TypeScript single-page application. Built with Vite, styled with Tailwind CSS, animated with Framer Motion. In development, Vite runs as Express middleware with HMR. In production, the client is compiled to `dist/public/` and served as static files by the Express server.

## File Structure

```
client/
├── index.html                          # Vite entry point
└── src/
    ├── main.tsx                        # React root render
    ├── App.tsx                         # Router setup (BrowserRouter, ProtectedRoute, PageGuard)
    ├── index.css                       # Tailwind imports + global styles
    ├── vite-env.d.ts                   # Vite client type declarations
    ├── api/
    │   └── client.ts                   # Axios instance, JWT interceptor, all API wrappers
    ├── components/
    │   ├── GameCanvas.tsx              # forwardRef HTML5 Canvas — draws ARC grids with 16-color palette
    │   ├── GamePreviewCanvas.tsx       # Static preview thumbnail (fetches initial frame from API)
    │   ├── VideoRecorder.tsx           # Server-streaming screen capture, record prompt, auto-stop
    │   ├── Layout.tsx                  # Admin sidebar with RBAC-filtered navigation
    │   ├── GameUploadForm.tsx          # Shared upload form (admin + request modes) with validation
    │   └── ConfirmModal.tsx            # Reusable confirm dialog (danger/warning/success variants)
    ├── hooks/
    │   ├── useAuth.tsx                 # AuthContext — login, logout, user state, token persistence
    │   ├── useGameEngine.tsx           # Admin game session management (start, action, end, reset)
    │   └── useVideoRecorder.ts         # Server-streaming recorder (720p, 4Mbps, zero client RAM)
    ├── pages/
    │   ├── HomePage.tsx                # Public game catalog with stats
    │   ├── PublicPlayPage.tsx           # Public play — auto-start, keyboard controls, animations
    │   ├── DirectPlayPage.tsx           # Ephemeral play — upload game.py + metadata.json, play immediately
    │   ├── LoginPage.tsx               # Admin login form
    │   ├── DashboardPage.tsx           # Admin dashboard — stats, charts, recent games
    │   ├── GamesPage.tsx               # Admin game list — search, filter, toggle, delete
    │   ├── GameDetailPage.tsx          # Admin game detail — metadata editor, analytics, source viewer
    │   ├── GameUploadPage.tsx          # Admin upload — uses GameUploadForm
    │   ├── GamePlayPage.tsx            # Admin play — authenticated game session with engine controls
    │   ├── UsersPage.tsx               # Admin user management — create, edit, delete, RBAC
    │   ├── RequestedGamesPage.tsx      # Admin request review — approve/reject with source preview
    │   └── TempGamesPage.tsx           # Admin temp session viewer — ephemeral session logs
    └── types/
        └── index.ts                    # TypeScript interfaces for all API responses and components
```

## Routes

### Public Routes (no auth)

| Path | Page | Description |
|------|------|-------------|
| `/` | `HomePage` | Game catalog, platform stats, featured game |
| `/play/:gameId` | `PublicPlayPage` | Play a game (auto-starts, supports `?name=` query param) |
| `/play-direct` | `DirectPlayPage` | Upload game files and play immediately (ephemeral session) |

### Admin Routes (auth required)

All admin routes are nested under `/admin` and wrapped in `<ProtectedRoute>` (redirects to login if no JWT). Individual pages are wrapped in `<PageGuard page="...">` for RBAC.

| Path | Page | Required Page |
|------|------|---------------|
| `/admin/login` | `LoginPage` | — (public) |
| `/admin` | `DashboardPage` | `dashboard` |
| `/admin/games` | `GamesPage` | `games` |
| `/admin/games/upload` | `GameUploadPage` | `upload` |
| `/admin/games/:gameId` | `GameDetailPage` | `games` |
| `/admin/games/:gameId/play` | `GamePlayPage` | `games` |
| `/admin/requests` | `RequestedGamesPage` | `requests` |
| `/admin/temp-games` | `TempGamesPage` | `games` |
| `/admin/users` | `UsersPage` | `users` |

Admin users bypass `PageGuard` entirely (always have full access).

## Key Components

### `GameCanvas`
`forwardRef` HTML5 Canvas component that renders ARC grids.

- Draws a 2D `number[][]` grid using the 16-color ARC palette
- Auto-computes cell size to fit within `maxCanvasWidth` x `maxCanvasHeight` (default 640x640)
- Supports click-to-cell coordinate mapping (`onCellClick(x, y)`)
- Optional grid lines (when `showGrid=true` and cell size > 8px)
- Uses `imageRendering: 'pixelated'` for crisp pixel art scaling
- Exposes the raw `<canvas>` element via `useImperativeHandle` for parent access

### `VideoRecorder`
Server-streaming screen capture component. **Zero client RAM usage** — chunks are streamed to the server in real-time and discarded immediately.

- **Record prompt**: shown when game starts (`isPlaying` transitions to true). Player chooses "Record" or "Skip". Recording unavailable for ephemeral games.
- **Streaming architecture**: every 2-second chunk is POST'd to `/api/games/public/:gameId/video/chunk` and discarded from client memory. Server appends to disk. Client RAM stays < 5 MB.
- **Codec priority**: MP4 (H.264) preferred, WebM (VP9/VP8) fallback.
- **Quality**: 720p, 30fps, 4 Mbps — excellent for grid games, low resource usage.
- **Auto-stop**: when `isGameOver` becomes true, recording stops after 1.5s delay. Also auto-stops at 5-minute max duration.
- **After recording**: shows "Saved" indicator with download link (from server) and delete button. No local preview needed since video is on the server.
- **Server upload flow**: `POST /video/start` → `POST /video/chunk` (repeated) → `POST /video/end`

### `Layout`
Admin sidebar with navigation sections (Overview, Games, Team).

- Filters nav items based on `user.allowed_pages` RBAC array
- Collapsible sidebar with animated width transition (Framer Motion)
- Active route indicator bar
- Tooltip labels when collapsed
- User info panel with logout button

### `GameUploadForm`
Shared form component used by both admin upload and public game submission.

- Two modes: `admin` (direct upload) and `request` (community submission)
- Drag-and-drop file zones for `game.py` and `metadata.json`
- Client-side validation:
  - `game.py`: rejects files containing comments (`#`) or docstrings (`"""`)
  - `metadata.json`: validates schema (allowed keys: `game_id`, `default_fps`, `baseline_actions`, `tags`, `local_dir`)
- Optional fields: game name, description, rules, owner, drive link, video link

### `ConfirmModal`
Reusable confirm dialog with three variants:

| Variant | Icon | Button Color |
|---------|------|-------------|
| `danger` | AlertTriangle (red) | Red |
| `warning` | AlertTriangle (yellow) | Yellow |
| `success` | CheckCircle (green) | Green |

Uses Framer Motion for enter/exit animations (fade + scale spring).

### `GamePreviewCanvas`
Static preview thumbnail that fetches the initial grid frame from `/api/games/public/:gameId/preview` and renders it on a small canvas. Shows a loading skeleton while fetching and a "No preview" placeholder on failure.

## Key Hooks

### `useAuth`
Context-based authentication hook.

- **State**: `{ user, loading, login, logout }`
- **Token storage**: `localStorage` key `arc_token`
- **Initialization**: on mount, if a token exists, calls `GET /api/auth/me` to validate and hydrate user state
- **Login flow**: `POST /api/auth/login` → stores token → `GET /api/auth/me` → sets user
- **Logout**: removes token from localStorage, clears user state

### `useGameEngine`
Admin game session management hook.

- **State**: `{ sessionGuid, frame, loading, error, isPlaying, isGameOver }`
- **`startGame(gameId, seed, startLevel)`**: calls `POST /api/player/start`, stores session GUID
- **`sendAction(action, x, y)`**: calls `POST /api/player/action/:guid`, updates frame. Detects session loss (404 / "No game instance") and clears state.
- **`endGame()`**: calls `POST /api/player/end/:guid`, clears all state
- **`resetGame()`**: sends RESET action via the action endpoint
- Exports `ARC_COLORS` — the 16-color hex palette as a `Record<number, string>`

### `useVideoRecorder`
Server-streaming screen capture hook. Streams 2-second video chunks directly to the server — no client-side accumulation.

- **State**: `{ isRecording, recordingTime, hasRecording, serverFilename, fileExt }`
- **Codec selection**: `pickCodec()` probes `MediaRecorder.isTypeSupported()` — tries MP4 (H.264 Baseline, generic H.264, plain MP4) first, then WebM (VP9, VP8, plain WebM)
- **`startRecording(gameId, playerName?)`**: calls `getDisplayMedia` at 720p/30fps, creates MediaRecorder at 4 Mbps. Calls `POST /video/start` to get a `recording_id`. Each 2s chunk is POST'd to `/video/chunk` and discarded.
- **`stopRecording()`**: stops recorder, waits for pending chunk uploads, calls `POST /video/end` to finalize
- **Max duration**: 5 minutes (auto-stops)
- **Memory**: < 5 MB constant regardless of recording length (chunks are fire-and-forget)
- **Cleanup**: on unmount, stops recorder and stream tracks

## API Client

Defined in `client/src/api/client.ts`. Uses Axios.

### Base Configuration
- Base URL: `/api` (same-origin — no proxy needed since Express serves both API and SPA)
- Default header: `Content-Type: application/json`

### Request Interceptor
Attaches JWT from `localStorage.getItem('arc_token')` as `Authorization: Bearer <token>` on every request.

### Response Interceptor
On 401 responses:
1. Removes `arc_token` from localStorage
2. If the current path starts with `/admin`, redirects to `/admin/login`
3. Public pages (e.g., `/play/:gameId`) do not redirect — they continue operating without auth

### API Namespaces

| Namespace | Methods |
|-----------|---------|
| `authAPI` | `login`, `me`, `register` |
| `gamesAPI` | `listPublic`, `getPublic`, `getPublicPlays`, `getPublicPreview`, `getPublicStats`, `getPublicGameStats`, `list`, `get`, `upload`, `update`, `updateFiles`, `toggle`, `delete`, `clearSessions`, `getSource`, `syncLocal`, `uploadVideo`, `listVideos`, `getVideoUrl`, `deleteVideo` |
| `playerAPI` | `publicStart`, `publicAction`, `publicEnd`, `ephemeralStart`, `ephemeralAction`, `ephemeralEnd`, `start`, `action`, `frame`, `end`, `palette` |
| `analyticsAPI` | `dashboard`, `game`, `sessions`, `replay`, `tempSessions`, `deleteTempSessions`, `exportExcel`, `exportAllExcel`, `exportGamesList` |
| `usersAPI` | `list`, `get`, `update`, `delete`, `changeProtectedPassword` |
| `requestsAPI` | `submit`, `list`, `get`, `getSource`, `review`, `delete`, `getFile` |

## Game Controls

### Keyboard Mapping

| Key | Action | Description |
|-----|--------|-------------|
| `W` / `ArrowUp` | `ACTION1` | Move up |
| `S` / `ArrowDown` | `ACTION2` | Move down |
| `A` / `ArrowLeft` | `ACTION3` | Move left |
| `D` / `ArrowRight` | `ACTION4` | Move right |
| `Space` / `F` | `ACTION5` | Primary action (interact/confirm) |
| `Click` on canvas | `ACTION6` | Coordinate action (sends `x`, `y`) |
| `Ctrl+Z` / `Cmd+Z` | `ACTION7` | Undo |
| `R` | `RESET` | Reset current level (double-tap = restart from level 1) |

Actions are throttled to 50ms minimum interval. Keys are only processed when the action is in the frame's `available_actions` array. When the game is over (`WIN` or `GAME_OVER`), only `R` (reset) is accepted.

## Animations

### Level Clear — Glitch/Digital TV Effect (2s)

A layered CSS animation triggered when `frame.level` increases:

1. **Scan lines** — horizontal repeating gradient lines (cyan-tinted, 2px spacing)
2. **RGB shift** — `box-shadow` insets that alternate red/cyan/blue/magenta offsets (0.4s, stepped)
3. **Horizontal glitch bars** — 6 randomly-positioned white bars that jitter left/right (0.1-0.25s each)
4. **Static noise** — SVG fractalNoise filter flashing at 30-60% opacity
5. **Blackout** — full black overlay that snaps on at 16%, flickers, holds 25-75%, then fades out
6. **Level text** — appears during blackout with RGB-split text effect (`clip-path` + `translate`). Shows "LEVEL N", move count, time, and a blinking "Loading level N+1_" prompt

Auto-dismisses after 2 seconds via `setTimeout`.

### Game Over — Skull + Shake

- **Canvas shake**: `translateX` oscillation (-4px/+4px) over 0.5s
- **Overlay**: radial gradient (red-tinted center to black edges)
- **Skull icon**: scales from 1.5x to 1x over 0.4s (ease-out)
- **Title**: "Game Over" fades up with 0.2s delay
- **Stats**: total time and actions displayed in bordered cards

### Win — Trophy + Sparkle

- **Overlay**: radial gradient (emerald-tinted center to black edges)
- **Trophy icon**: scales from 0 with -30deg rotation, overshoots to 1.2x, settles at 1x (0.6s spring)
- **Pulsing ring**: border scales 1x-1.08x in 2s loop
- **Sparkle particles**: 20 randomly-positioned colored dots (emerald, yellow, blue, pink, purple) that scale up, float upward, and fade out in 2.5s infinite loops
- **Title**: "Victory!" with gradient text (`emerald → teal → cyan`), fades up with 0.3s delay
- **Stats**: total time and actions in emerald-bordered cards, fade up with 0.5s delay
- **CTA button**: "Back to Games" with emerald gradient, fades up with 0.7s delay

### Reset — Fade + Scale

- Canvas wrapper transitions `opacity: 0` and `scale: 0.95` for 400ms, then restores
- Reset button icon plays `animate-spin` during the animation

## Color Palette Reference

The ARC-AGI-3 16-color palette used by `GameCanvas` and `GamePreviewCanvas`:

| Index | Hex | Color | Name |
|-------|-----|-------|------|
| 0 | `#FFFFFF` | ![#FFFFFF](https://placehold.co/16x16/FFFFFF/FFFFFF) | White |
| 1 | `#CCCCCC` | ![#CCCCCC](https://placehold.co/16x16/CCCCCC/CCCCCC) | Off-white |
| 2 | `#999999` | ![#999999](https://placehold.co/16x16/999999/999999) | Light Grey |
| 3 | `#666666` | ![#666666](https://placehold.co/16x16/666666/666666) | Grey |
| 4 | `#333333` | ![#333333](https://placehold.co/16x16/333333/333333) | Dark Grey |
| 5 | `#000000` | ![#000000](https://placehold.co/16x16/000000/000000) | Black |
| 6 | `#E53AA3` | ![#E53AA3](https://placehold.co/16x16/E53AA3/E53AA3) | Magenta |
| 7 | `#FF7BCC` | ![#FF7BCC](https://placehold.co/16x16/FF7BCC/FF7BCC) | Pink |
| 8 | `#F93C31` | ![#F93C31](https://placehold.co/16x16/F93C31/F93C31) | Red |
| 9 | `#1E93FF` | ![#1E93FF](https://placehold.co/16x16/1E93FF/1E93FF) | Blue |
| 10 | `#88D8F1` | ![#88D8F1](https://placehold.co/16x16/88D8F1/88D8F1) | Light Blue |
| 11 | `#FFDC00` | ![#FFDC00](https://placehold.co/16x16/FFDC00/FFDC00) | Yellow |
| 12 | `#FF851B` | ![#FF851B](https://placehold.co/16x16/FF851B/FF851B) | Orange |
| 13 | `#921231` | ![#921231](https://placehold.co/16x16/921231/921231) | Maroon |
| 14 | `#4FCC30` | ![#4FCC30](https://placehold.co/16x16/4FCC30/4FCC30) | Green |
| 15 | `#A356D6` | ![#A356D6](https://placehold.co/16x16/A356D6/A356D6) | Purple |

Note: The Python `game_runner.py` uses a different 10-color palette (indices 0-9) for the ARC engine itself. The 16-color palette above is the client-side rendering palette defined in `GameCanvas.tsx` and `useGameEngine.tsx`.
