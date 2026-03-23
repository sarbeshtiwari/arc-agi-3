import uuid
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist():
    return datetime.now(IST).replace(tzinfo=None)

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    allowed_pages = Column(JSON, default=list)  # e.g. ["dashboard","games","upload","requests","users"]
    created_at = Column(DateTime, default=now_ist)
    updated_at = Column(DateTime, default=now_ist, onupdate=now_ist)

    play_sessions = relationship("PlaySession", back_populates="user")


class Game(Base):
    __tablename__ = "games"

    id = Column(String, primary_key=True, default=generate_uuid)
    game_id = Column(String(50), unique=True, nullable=False, index=True)  # e.g. "ls20-v1"
    name = Column(String(100), nullable=False)  # Human-readable name
    description = Column(Text, nullable=True)
    game_rules = Column(Text, nullable=True)  # How to play / rules
    game_owner_name = Column(String(100), nullable=True)
    game_drive_link = Column(String(500), nullable=True)
    game_video_link = Column(String(500), nullable=True)
    version = Column(String(20), nullable=False, default="v1")
    game_code = Column(String(10), nullable=False)  # 4-char code, e.g. "ls20"

    # Status
    is_active = Column(Boolean, default=True)

    # Metadata from metadata.json
    default_fps = Column(Integer, default=5)
    baseline_actions = Column(JSON, nullable=True)  # list of avg actions per level
    tags = Column(JSON, nullable=True)  # list of tags
    grid_max_size = Column(Integer, default=64)

    # File paths (relative to environment_files dir)
    game_file_path = Column(String(500), nullable=False)  # path to .py file
    metadata_file_path = Column(String(500), nullable=False)  # path to metadata.json
    local_dir = Column(String(500), nullable=False)  # directory containing game files

    # Stats
    total_plays = Column(Integer, default=0)
    total_wins = Column(Integer, default=0)
    avg_score = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=now_ist)
    updated_at = Column(DateTime, default=now_ist, onupdate=now_ist)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=True)

    play_sessions = relationship("PlaySession", back_populates="game", cascade="all, delete-orphan")
    analytics = relationship("GameAnalytics", cascade="all, delete-orphan")
    uploader = relationship("User", foreign_keys=[uploaded_by])


class PlaySession(Base):
    __tablename__ = "play_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    game_id = Column(String, ForeignKey("games.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    session_guid = Column(String(100), unique=True, nullable=False, default=generate_uuid)

    # Game state
    seed = Column(Integer, default=0)
    current_level = Column(Integer, default=0)
    state = Column(String(20), default="NOT_FINISHED")  # NOT_FINISHED, WIN, GAME_OVER
    score = Column(Float, default=0.0)
    total_actions = Column(Integer, default=0)
    game_overs = Column(Integer, default=0)  # number of times player hit GAME_OVER

    # Player info (for public/anonymous players)
    player_name = Column(String(100), nullable=True)

    # Timing
    total_time = Column(Float, default=0.0)  # total seconds

    # Per-level stats: [{level, actions, time, completed}, ...]
    level_stats = Column(JSON, default=list)

    # Action replay log
    action_log = Column(JSON, default=list)  # [{action, data, timestamp, frame_idx}, ...]

    # Timestamps
    started_at = Column(DateTime, default=now_ist)
    ended_at = Column(DateTime, nullable=True)

    game = relationship("Game", back_populates="play_sessions")
    user = relationship("User", back_populates="play_sessions")


class GameAnalytics(Base):
    __tablename__ = "game_analytics"

    id = Column(String, primary_key=True, default=generate_uuid)
    game_id = Column(String, ForeignKey("games.id"), nullable=False)
    date = Column(DateTime, default=now_ist)

    # Daily aggregates
    plays_count = Column(Integer, default=0)
    wins_count = Column(Integer, default=0)
    avg_actions_to_win = Column(Float, default=0.0)
    avg_score = Column(Float, default=0.0)
    unique_players = Column(Integer, default=0)

    game = relationship("Game")


class GameRequest(Base):
    """Pending game submission from public users, awaiting admin approval."""
    __tablename__ = "game_requests"

    id = Column(String, primary_key=True, default=generate_uuid)
    game_id = Column(String(50), unique=True, nullable=False, index=True)
    requester_name = Column(String(100), nullable=False)
    requester_email = Column(String(255), nullable=True)
    message = Column(Text, nullable=True)  # optional note from requester
    description = Column(Text, nullable=True)
    game_rules = Column(Text, nullable=True)  # How to play / rules
    game_owner_name = Column(String(100), nullable=True)
    game_drive_link = Column(String(500), nullable=True)
    game_video_link = Column(String(500), nullable=True)

    # Status: pending, approved, rejected
    status = Column(String(20), default="pending", index=True)
    admin_note = Column(Text, nullable=True)  # optional note from admin

    # File storage (stored in DB as binary + on disk as backup)
    game_file_path = Column(String(500), nullable=True)
    metadata_file_path = Column(String(500), nullable=True)
    local_dir = Column(String(500), nullable=True)
    game_file_content = Column(LargeBinary, nullable=True)  # .py file bytes
    metadata_file_content = Column(LargeBinary, nullable=True)  # metadata.json bytes

    # Parsed metadata
    game_code = Column(String(50), nullable=True)
    version = Column(String(20), default="v1")
    tags = Column(JSON, nullable=True)
    default_fps = Column(Integer, default=5)

    # Timestamps
    created_at = Column(DateTime, default=now_ist)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, ForeignKey("users.id"), nullable=True)


class TempGameSession(Base):
    """Logs ephemeral 'Play Your Own' sessions. Auto-deleted after 24 hours."""
    __tablename__ = "temp_game_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_guid = Column(String(50), unique=True, nullable=False, index=True)
    game_id = Column(String(50), nullable=True)
    player_name = Column(String(100), nullable=True)

    # Game state
    state = Column(String(20), default="NOT_FINISHED")
    score = Column(Float, default=0.0)
    total_actions = Column(Integer, default=0)
    current_level = Column(Integer, default=0)
    game_overs = Column(Integer, default=0)
    total_time = Column(Float, default=0.0)
    level_stats = Column(JSON, nullable=True)

    # Action log (list of {action, timestamp, level})
    action_log = Column(JSON, default=list)

    # Timestamps
    started_at = Column(DateTime, default=now_ist)
    ended_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)  # started_at + 24hrs
