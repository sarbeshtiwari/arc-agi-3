// ──── Role & Approval Types ────

export type UserRole = 'tasker' | 'ql' | 'pl' | 'super_admin';

export type ApprovalStatus =
  | 'draft'
  | 'pending_ql'
  | 'ql_approved'
  | 'pending_pl'
  | 'approved'
  | 'rejected';

export const ROLE_LABELS: Record<UserRole, string> = {
  tasker: 'Tasker',
  ql: 'Quality Lead',
  pl: 'Project Lead',
  super_admin: 'Super Admin',
};

export const APPROVAL_STATUS_LABELS: Record<ApprovalStatus, string> = {
  draft: 'Draft',
  pending_ql: 'Pending QL Review',
  ql_approved: 'QL Approved',
  pending_pl: 'Pending PL Review',
  approved: 'Approved',
  rejected: 'Rejected',
};

// ──── API Response Types ────

export interface User {
  id: string;
  username: string;
  display_name: string | null;
  email: string | null;
  is_admin: boolean;
  is_active: boolean;
  role: UserRole;
  team_lead_id: string | null;
  team_lead_ids: string[];
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
  approval_status: ApprovalStatus;
  assigned_ql_id: string | null;
  assigned_pl_id: string | null;
  rejection_reason: string | null;
  rejection_by: string | null;
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

// ──── API Request Bodies ────

export interface CreateUserRequest {
  username: string;
  password: string;
  display_name?: string;
  email?: string;
  is_admin?: boolean;
  role?: UserRole;
  team_lead_ids?: string[];
  allowed_pages?: string[];
}

export interface UpdateUserRequest {
  email?: string;
  display_name?: string;
  is_admin?: boolean;
  is_active?: boolean;
  password?: string;
  role?: UserRole;
  team_lead_ids?: string[];
  allowed_pages?: string[];
}

export interface UpdateGameRequest {
  name?: string;
  description?: string;
  game_rules?: string;
  game_owner_name?: string;
  game_drive_link?: string;
  game_video_link?: string;
  is_active?: boolean;
  default_fps?: number;
  tags?: string[];
}

export interface GameToggleRequest {
  is_active: boolean;
}

export interface PlaySessionStartRequest {
  game_id: string;
  seed?: number;
  player_name?: string;
  start_level?: number;
}

export interface GameActionRequest {
  action: string;
  x?: number;
  y?: number;
}

export interface ReviewRequestBody {
  action: "approve" | "reject";
  admin_note?: string;
}

// ──── ARC Color Palette ────

export const ARC_COLORS: Record<number, string> = {
  0: '#FFFFFF',   // White
  1: '#CCCCCC',   // Off-white
  2: '#999999',   // Light Grey
  3: '#666666',   // Grey
  4: '#333333',   // Dark Grey
  5: '#000000',   // Black
  6: '#E53AA3',   // Magenta
  7: '#FF7BCC',   // Pink
  8: '#F93C31',   // Red
  9: '#1E93FF',   // Blue
  10: '#88D8F1',  // Light Blue
  11: '#FFDC00',  // Yellow
  12: '#FF851B',  // Orange
  13: '#921231',  // Maroon
  14: '#4FCC30',  // Green
  15: '#A356D6',  // Purple
};

// ──── Notification Types ────

export type NotificationType =
  | 'game_submitted'
  | 'game_approved_ql'
  | 'game_approved_pl'
  | 'game_rejected'
  | 'game_resubmitted'
  | 'team_assigned'
  | 'game_updated';

export interface Notification {
  id: string;
  user_id: string;
  type: NotificationType;
  title: string;
  message: string;
  game_id: string | null;
  is_read: boolean;
  created_at: string;
}

// ──── Audit Log Types ────

export type AuditAction =
  | 'game_uploaded'
  | 'game_updated'
  | 'game_submitted_for_review'
  | 'game_approved_by_ql'
  | 'game_rejected_by_ql'
  | 'game_approved_by_pl'
  | 'game_rejected_by_pl'
  | 'game_approved_by_admin'
  | 'game_version_updated'
  | 'game_resubmitted';

export interface AuditLogEntry {
  id: string;
  game_id: string;
  user_id: string;
  username?: string;
  action: AuditAction;
  details: Record<string, any> | null;
  created_at: string;
}

// ──── Approval Workflow Request Types ────

export interface SubmitForReviewRequest {
  message?: string;
}

export interface ApprovalActionRequest {
  action: 'approve' | 'reject';
  reason?: string;
}

// ──── Team Types ────

export interface TeamMember {
  id: string;
  username: string;
  display_name: string | null;
  email: string | null;
  role: UserRole;
  is_active: boolean;
  team_lead_id: string | null;
  team_lead_ids: string[];
  team_lead_usernames: string[];
  created_at: string;
}

export interface TeamAssignment {
  user_id: string;
  lead_ids: string[];
}
