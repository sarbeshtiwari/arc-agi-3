// ──── API Response Types ────

export interface User {
  id: string;
  username: string;
  email: string | null;
  is_admin: boolean;
  is_active: boolean;
  allowed_pages: string[];
  created_at: string;
}

export interface Game {
  id: string;
  game_id: string;
  name: string;
  description: string | null;
  game_rules: string | null;
  game_owner_name: string | null;
  game_drive_link: string | null;
  game_video_link: string | null;
  version: string;
  game_code: string;
  is_active: boolean;
  default_fps: number;
  baseline_actions: number[] | null;
  tags: string[] | null;
  grid_max_size: number;
  total_plays: number;
  total_wins: number;
  created_at: string;
  updated_at: string;
}

export interface PlaySession {
  id: string;
  session_guid: string;
  game_id: string;
  seed: number;
  current_level: number;
  state: GameState;
  score: number;
  total_actions: number;
  total_time: number;
  level_stats: LevelStat[];
  player_name: string | null;
  game_overs: number;
  started_at: string;
  ended_at: string | null;
}

export interface GameRequest {
  id: string;
  game_id: string;
  requester_name: string;
  requester_email: string | null;
  message: string | null;
  description: string | null;
  game_rules: string | null;
  game_owner_name: string | null;
  game_drive_link: string | null;
  game_video_link: string | null;
  status: 'pending' | 'approved' | 'rejected';
  admin_note: string | null;
  game_code: string | null;
  version: string;
  tags: string[] | null;
  default_fps: number;
  created_at: string;
  reviewed_at: string | null;
}

export interface TempGameSession {
  id: string;
  session_guid: string;
  game_id: string | null;
  player_name: string | null;
  state: GameState;
  score: number;
  total_actions: number;
  current_level: number;
  game_overs: number;
  total_time: number;
  level_stats: LevelStat[] | null;
  action_log: ActionLogEntry[];
  started_at: string;
  ended_at: string | null;
  expires_at: string;
}

// ──── Game Engine Types ────

export type GameState = 'NOT_FINISHED' | 'WIN' | 'GAME_OVER';

export type GameAction =
  | 'ACTION1' | 'ACTION2' | 'ACTION3' | 'ACTION4'
  | 'ACTION5' | 'ACTION6' | 'ACTION7' | 'RESET';

export interface GameFrame {
  grid: number[][];
  width: number;
  height: number;
  state: GameState;
  score: number;
  level: number;
  total_actions: number;
  available_actions: string[];
  metadata: FrameMetadata | null;
}

export interface FrameMetadata {
  session_guid?: string;
  seed?: number;
  game_id?: string;
  mode?: string;
  total_time?: number;
  current_level_stats?: LevelStat | null;
  completed_levels?: LevelStat[];
  ephemeral?: boolean;
  total_levels?: number;
  game_overs?: number;
  [key: string]: any;
}

export interface LevelStat {
  level: number;
  actions: number;
  time: number;
  completed: boolean;
  lives_used?: number;
  game_overs?: number;
  resets?: number;
}

export interface ActionLogEntry {
  action: string;
  timestamp?: number;
  level?: number;
  x?: number;
  y?: number;
}

// ──── Analytics Types ────

export interface DashboardStats {
  total_games: number;
  active_games: number;
  total_plays: number;
  total_users: number;
  total_wins: number;
  total_game_overs: number;
  win_rate: number;
  pending_requests: number;
  daily_plays: number[];
  daily_wins: number[];
  daily_labels: string[];
  game_distribution: Array<{ name: string; plays: number }>;
  recent_games: Game[];
  top_played_games: Array<{
    game_id: string;
    name: string;
    total_plays: number;
    total_wins: number;
  }>;
}

export interface GameAnalytics {
  game_id: string;
  game_name: string;
  total_plays: number;
  total_wins: number;
  win_rate: number;
  avg_actions_to_win: number;
  avg_time_to_win: number;
  unique_players: number;
  daily_stats: Array<{
    date: string;
    plays: number;
    wins: number;
    avg_time: number;
  }>;
}

export interface SessionSummary {
  id: string;
  game_name: string;
  game_id: string;
  player: string;
  state: GameState;
  total_actions: number;
  total_time: number;
  level_stats: LevelStat[];
  current_level: number;
  game_overs?: number;
  score?: number;
  started_at: string;
}

// ──── Auth Types ────

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// ──── Component Prop Types ────

export interface GameCanvasProps {
  grid: number[][];
  width: number;
  height: number;
  onCellClick?: (x: number, y: number) => void;
  cellSize?: number;
  showGrid?: boolean;
  maxCanvasWidth?: number;
  maxCanvasHeight?: number;
  interactive?: boolean;
}

export interface ConfirmModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'danger' | 'warning' | 'success';
  loading?: boolean;
}

export interface LevelClearAnim {
  level: number;
  nextLevel: number;
  actions: number;
  time: number;
}
