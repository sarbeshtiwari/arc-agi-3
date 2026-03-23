from datetime import datetime
from pydantic import BaseModel, Field


# ──── Auth ────
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: str | None = None
    is_admin: bool = False
    allowed_pages: list[str] = []  # e.g. ["dashboard","games","upload","requests","users"]


class UserResponse(BaseModel):
    id: str
    username: str
    email: str | None
    is_admin: bool
    is_active: bool
    allowed_pages: list | None
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    email: str | None = None
    is_admin: bool | None = None
    is_active: bool | None = None
    allowed_pages: list[str] | None = None
    password: str | None = None


# ──── Game ────
class GameResponse(BaseModel):
    id: str
    game_id: str
    name: str
    description: str | None
    game_rules: str | None
    game_owner_name: str | None
    game_drive_link: str | None
    game_video_link: str | None
    version: str
    game_code: str
    is_active: bool
    default_fps: int
    baseline_actions: list | None
    tags: list | None
    grid_max_size: int
    total_plays: int
    total_wins: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GameUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    game_rules: str | None = None
    game_owner_name: str | None = None
    game_drive_link: str | None = None
    game_video_link: str | None = None
    is_active: bool | None = None
    default_fps: int | None = None
    tags: list[str] | None = None


class GameToggleRequest(BaseModel):
    is_active: bool


# ──── Play Session ────
class PlaySessionCreate(BaseModel):
    game_id: str
    seed: int = 0
    player_name: str | None = None
    start_level: int = 0  # level to start from (0-indexed)


class PlaySessionResponse(BaseModel):
    id: str
    session_guid: str
    game_id: str
    seed: int
    current_level: int
    state: str
    score: float
    total_actions: int
    started_at: datetime
    ended_at: datetime | None

    class Config:
        from_attributes = True


class GameActionRequest(BaseModel):
    action: str  # ACTION1-7, RESET
    x: int | None = None
    y: int | None = None


class GameFrameResponse(BaseModel):
    grid: list[list[int]]  # 2D array of color values
    width: int
    height: int
    state: str  # NOT_FINISHED, WIN, GAME_OVER
    score: float
    level: int
    total_actions: int
    available_actions: list[str]
    metadata: dict | None = None


# ──── Analytics ────
class GameAnalyticsResponse(BaseModel):
    game_id: str
    game_name: str
    total_plays: int
    total_wins: int
    win_rate: float
    avg_actions_to_win: float
    avg_time_to_win: float
    daily_stats: list[dict]


class DashboardStats(BaseModel):
    total_games: int
    active_games: int
    total_plays: int
    total_users: int
    total_wins: int = 0
    total_game_overs: int = 0
    pending_requests: int = 0
    win_rate: float = 0
    daily_plays: list[int] = []
    daily_wins: list[int] = []
    daily_labels: list[str] = []
    game_distribution: list[dict] = []
    recent_games: list[GameResponse]
    top_played_games: list[dict]


# ──── Game Requests ────
class GameRequestResponse(BaseModel):
    id: str
    game_id: str
    requester_name: str
    requester_email: str | None
    message: str | None
    description: str | None
    game_rules: str | None
    game_owner_name: str | None
    game_drive_link: str | None
    game_video_link: str | None
    status: str
    admin_note: str | None
    game_code: str | None
    version: str
    tags: list | None
    default_fps: int
    created_at: datetime
    reviewed_at: datetime | None

    class Config:
        from_attributes = True


class GameRequestReview(BaseModel):
    action: str  # "approve" or "reject"
    admin_note: str | None = None
