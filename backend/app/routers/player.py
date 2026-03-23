import uuid
import os
import json
import shutil
import tempfile
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

logger = logging.getLogger("arc-agi.player")

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Game, PlaySession, User
from app.schemas import (
    GameActionRequest,
    GameFrameResponse,
    PlaySessionCreate,
    PlaySessionResponse,
)
from app.services.analytics import AnalyticsService
from app.services.game_engine import GameEngineService

router = APIRouter(prefix="/api/player", tags=["player"])

# Global engine service instance
_engine_service: GameEngineService | None = None


def get_engine() -> GameEngineService:
    global _engine_service
    if _engine_service is None:
        _engine_service = GameEngineService(settings.ENVIRONMENT_FILES_DIR)
    return _engine_service


# ──────────────────────────────────────────────
# Public endpoints (no auth) for guest players
# ──────────────────────────────────────────────

@router.post("/public/start", response_model=GameFrameResponse)
def public_start_session(
    request: PlaySessionCreate,
    db: Session = Depends(get_db),
    engine: GameEngineService = Depends(get_engine),
):
    game = db.query(Game).filter(Game.game_id == request.game_id, Game.is_active == True).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found or inactive")

    session_guid = str(uuid.uuid4())
    play_session = PlaySession(
        game_id=game.id,
        user_id=None,
        session_guid=session_guid,
        seed=request.seed,
        player_name=request.player_name or None,
    )
    db.add(play_session)

    try:
        engine.create_instance(
            session_id=session_guid,
            game_id=game.game_id,
            game_code=game.game_code,
            game_dir=game.local_dir,
            seed=request.seed,
        )
        # Skip to requested level if specified
        if request.start_level and request.start_level > 0:
            engine.skip_to_level(session_guid, request.start_level)
    except Exception as e:
        db.expunge(play_session)
        raise HTTPException(status_code=500, detail=f"Failed to start game: {e}")

    db.commit()
    db.refresh(play_session)

    frame = engine.get_frame(session_guid)
    return GameFrameResponse(
        grid=frame.grid,
        width=frame.width,
        height=frame.height,
        state=frame.state,
        score=frame.score,
        level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid, **(frame.metadata or {})},
    )


@router.post("/public/action/{session_guid}", response_model=GameFrameResponse)
def public_take_action(
    session_guid: str,
    request: GameActionRequest,
    db: Session = Depends(get_db),
    engine: GameEngineService = Depends(get_engine),
):
    play_session = (
        db.query(PlaySession)
        .filter(PlaySession.session_guid == session_guid)
        .first()
    )
    if not play_session:
        raise HTTPException(status_code=404, detail="Play session not found")

    if play_session.state == "WIN":
        try:
            frame = engine.get_frame(session_guid)
            return GameFrameResponse(
                grid=frame.grid, width=frame.width, height=frame.height,
                state=frame.state, score=frame.score, level=frame.level,
                total_actions=frame.total_actions,
                available_actions=frame.available_actions,
                metadata={"session_guid": session_guid},
            )
        except Exception:
            return GameFrameResponse(
                grid=[[0]], width=1, height=1,
                state=str(play_session.state),
                score=float(play_session.score or 0),
                level=int(play_session.current_level or 0),
                total_actions=int(play_session.total_actions or 0),
                available_actions=[],
                metadata={"session_guid": session_guid},
            )

    try:
        frame = engine.step(
            session_id=session_guid, action=request.action,
            x=request.x, y=request.y,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Action failed for session {session_guid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Action failed: {e}")

    play_session.state = frame.state
    play_session.total_actions = frame.total_actions
    play_session.current_level = frame.level

    meta = frame.metadata or {}
    if meta.get("total_time"):
        play_session.total_time = meta["total_time"]

    completed = meta.get("completed_levels")
    if completed and len(completed) != len(play_session.level_stats or []):
        play_session.level_stats = completed
        play_session.score = frame.score

    if meta.get("total_game_overs") is not None:
        play_session.game_overs = meta["total_game_overs"]

    if frame.state == "WIN":
        AnalyticsService.record_play_end(db, play_session)
    else:
        db.commit()

    return GameFrameResponse(
        grid=frame.grid, width=frame.width, height=frame.height,
        state=frame.state, score=frame.score, level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid, **(frame.metadata or {})},
    )


@router.post("/public/end/{session_guid}")
def public_end_session(
    session_guid: str,
    db: Session = Depends(get_db),
    engine: GameEngineService = Depends(get_engine),
):
    """End a play session (no auth required)."""
    play_session = (
        db.query(PlaySession)
        .filter(PlaySession.session_guid == session_guid)
        .first()
    )
    if play_session and play_session.state == "NOT_FINISHED":
        play_session.state = "GAME_OVER"
        try:
            frame = engine.get_frame(session_guid)
            meta = frame.metadata or {}
            if meta.get("total_time"):
                play_session.total_time = meta["total_time"]
            if meta.get("completed_levels"):
                play_session.level_stats = meta["completed_levels"]
        except Exception:
            pass
        AnalyticsService.record_play_end(db, play_session)

    try:
        engine.destroy_instance(session_guid)
    except Exception:
        pass

    return {"detail": "Session ended"}


# ──────────────────────────────────────────────
# Ephemeral / Direct play (logged to temp_game_sessions for 24hrs)
# ──────────────────────────────────────────────

# Track temp dirs so we can clean them up
_temp_dirs: dict[str, str] = {}


@router.post("/ephemeral/start", response_model=GameFrameResponse)
async def ephemeral_start(
    game_file: UploadFile = File(..., description="The game .py file"),
    metadata_file: UploadFile = File(..., description="The metadata.json file"),
    player_name: str = Form(""),
    engine: GameEngineService = Depends(get_engine),
    db: Session = Depends(get_db),
):
    """
    Start an ephemeral game session from uploaded files.
    Session is logged to temp_game_sessions DB table (auto-deleted after 24hrs).
    """
    from app.models import TempGameSession
    from datetime import timedelta

    game_py_bytes = await game_file.read()
    metadata_bytes = await metadata_file.read()

    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata.json")

    game_id = metadata.get("game_id", "temp")

    # Extract game_code from game_id
    if "-" in game_id:
        game_code = game_id.rsplit("-", 1)[0]
    else:
        game_code = game_id

    temp_base = tempfile.mkdtemp(prefix="arc_ephemeral_")
    game_dir = os.path.join(temp_base, game_code, "v1")
    os.makedirs(game_dir, exist_ok=True)

    game_py_path = os.path.join(game_dir, f"{game_code}.py")
    metadata_path = os.path.join(game_dir, "metadata.json")

    with open(game_py_path, "wb") as f:
        f.write(game_py_bytes)
    with open(metadata_path, "wb") as f:
        f.write(metadata_bytes)

    # Create engine instance
    session_guid = f"eph_{uuid.uuid4().hex[:12]}"
    try:
        engine.create_instance(
            session_id=session_guid,
            game_id=game_id,
            game_code=game_code,
            game_dir=game_dir,
            seed=0,
        )
    except Exception as e:
        shutil.rmtree(temp_base, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to start game: {e}")

    _temp_dirs[session_guid] = temp_base

    # Log to DB
    from app.models import now_ist
    now = now_ist()
    temp_session = TempGameSession(
        session_guid=session_guid,
        game_id=game_id,
        player_name=player_name.strip() or None,
        state="NOT_FINISHED",
        started_at=now,
        expires_at=now + timedelta(hours=24),
        action_log=[],
    )
    db.add(temp_session)
    db.commit()

    frame = engine.get_frame(session_guid)
    return GameFrameResponse(
        grid=frame.grid,
        width=frame.width,
        height=frame.height,
        state=frame.state,
        score=frame.score,
        level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid, "ephemeral": True, **(frame.metadata or {})},
    )


@router.post("/ephemeral/action/{session_guid}", response_model=GameFrameResponse)
def ephemeral_action(
    session_guid: str,
    request: GameActionRequest,
    engine: GameEngineService = Depends(get_engine),
    db: Session = Depends(get_db),
):
    """Execute an action in an ephemeral session and log it."""
    from app.models import TempGameSession

    try:
        frame = engine.step(
            session_id=session_guid,
            action=request.action,
            x=request.x,
            y=request.y,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action failed: {e}")

    # Update temp session in DB
    temp_session = db.query(TempGameSession).filter(
        TempGameSession.session_guid == session_guid
    ).first()
    if temp_session:
        temp_session.state = frame.state
        temp_session.score = frame.score
        temp_session.total_actions = frame.total_actions
        temp_session.current_level = frame.level

        meta = frame.metadata or {}
        if meta.get("total_time"):
            temp_session.total_time = meta["total_time"]
        if meta.get("completed_levels"):
            temp_session.level_stats = meta["completed_levels"]

        # Update session-level game_overs from engine total
        if meta.get("total_game_overs") is not None:
            temp_session.game_overs = meta["total_game_overs"]

        # Append to action log
        log = temp_session.action_log or []
        log.append({
            "action": request.action,
            "level": frame.level,
            "time": meta.get("total_time", 0),
        })
        temp_session.action_log = log

        db.commit()

    return GameFrameResponse(
        grid=frame.grid,
        width=frame.width,
        height=frame.height,
        state=frame.state,
        score=frame.score,
        level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid, "ephemeral": True, **(frame.metadata or {})},
    )


@router.post("/ephemeral/end/{session_guid}")
def ephemeral_end(
    session_guid: str,
    engine: GameEngineService = Depends(get_engine),
    db: Session = Depends(get_db),
):
    """End an ephemeral session and clean up temp files."""
    from app.models import TempGameSession, now_ist

    # Capture final state before destroying
    temp_session = db.query(TempGameSession).filter(
        TempGameSession.session_guid == session_guid
    ).first()
    if temp_session and temp_session.state == "NOT_FINISHED":
        try:
            frame = engine.get_frame(session_guid)
            temp_session.state = "GAME_OVER"
            meta = frame.metadata or {}
            if meta.get("total_time"):
                temp_session.total_time = meta["total_time"]
            if meta.get("completed_levels"):
                temp_session.level_stats = meta["completed_levels"]
            if meta.get("total_game_overs") is not None:
                temp_session.game_overs = meta["total_game_overs"]
        except Exception:
            temp_session.state = "GAME_OVER"
        temp_session.ended_at = now_ist()
        db.commit()

    try:
        engine.destroy_instance(session_guid)
    except Exception:
        pass

    # Clean up temp dir
    temp_dir = _temp_dirs.pop(session_guid, None)
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"detail": "Ephemeral session ended"}


# ──────────────────────────────────────────────
# Authenticated endpoints (admin player)
# ──────────────────────────────────────────────


# ──── Start Play Session ────
@router.post("/start", response_model=GameFrameResponse)
def start_session(
    request: PlaySessionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    engine: GameEngineService = Depends(get_engine),
):
    """Start a new play session for a game."""
    game = db.query(Game).filter(Game.game_id == request.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if not game.is_active:
        raise HTTPException(status_code=400, detail="Game is deactivated")

    # Create play session in DB
    session_guid = str(uuid.uuid4())
    play_session = PlaySession(
        game_id=game.id,
        user_id=user.id,
        session_guid=session_guid,
        seed=request.seed,
    )
    db.add(play_session)

    # Create engine instance
    try:
        instance = engine.create_instance(
            session_id=session_guid,
            game_id=game.game_id,
            game_code=game.game_code,
            game_dir=game.local_dir,
            seed=request.seed,
        )
        # Skip to requested level if specified
        if request.start_level and request.start_level > 0:
            engine.skip_to_level(session_guid, request.start_level)
    except Exception as e:
        db.expunge(play_session)
        raise HTTPException(status_code=500, detail=f"Failed to start game: {e}")

    db.commit()
    db.refresh(play_session)

    frame = engine.get_frame(session_guid)
    return GameFrameResponse(
        grid=frame.grid,
        width=frame.width,
        height=frame.height,
        state=frame.state,
        score=frame.score,
        level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid, **(frame.metadata or {})},
    )


# ──── Take Action ────
@router.post("/action/{session_guid}", response_model=GameFrameResponse)
def take_action(
    session_guid: str,
    request: GameActionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    engine: GameEngineService = Depends(get_engine),
):
    """Execute an action in an active play session."""
    # Verify session exists
    play_session = (
        db.query(PlaySession)
        .filter(PlaySession.session_guid == session_guid)
        .first()
    )
    if not play_session:
        raise HTTPException(status_code=404, detail="Play session not found")

    if play_session.state == "WIN":
        try:
            frame = engine.get_frame(session_guid)
            return GameFrameResponse(
                grid=frame.grid,
                width=frame.width,
                height=frame.height,
                state=frame.state,
                score=frame.score,
                level=frame.level,
                total_actions=frame.total_actions,
                available_actions=frame.available_actions,
                metadata={"session_guid": session_guid},
            )
        except Exception:
            return GameFrameResponse(
                grid=[[0]], width=1, height=1,
                state=str(play_session.state),
                score=float(play_session.score or 0),
                level=int(play_session.current_level or 0),
                total_actions=int(play_session.total_actions or 0),
                available_actions=[],
                metadata={"session_guid": session_guid},
            )

    # Execute action
    try:
        frame = engine.step(
            session_id=session_guid,
            action=request.action,
            x=request.x,
            y=request.y,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action failed: {e}")

    # Update session in DB
    play_session.state = frame.state
    play_session.score = frame.score
    play_session.total_actions = frame.total_actions
    play_session.current_level = frame.level

    # Store timing + level stats from metadata
    meta = frame.metadata or {}
    if meta.get("total_time"):
        play_session.total_time = meta["total_time"]
    if meta.get("completed_levels"):
        play_session.level_stats = meta["completed_levels"]

    # Update session-level game_overs from engine total
    if meta.get("total_game_overs") is not None:
        play_session.game_overs = meta["total_game_overs"]

    # If game ended, record analytics
    if frame.state in ("WIN", "GAME_OVER"):
        AnalyticsService.record_play_end(db, play_session)
    else:
        db.commit()

    return GameFrameResponse(
        grid=frame.grid,
        width=frame.width,
        height=frame.height,
        state=frame.state,
        score=frame.score,
        level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid, **(frame.metadata or {})},
    )


# ──── Get Current Frame ────
@router.get("/frame/{session_guid}", response_model=GameFrameResponse)
def get_current_frame(
    session_guid: str,
    user: User = Depends(get_current_user),
    engine: GameEngineService = Depends(get_engine),
):
    """Get the current frame of a play session."""
    try:
        frame = engine.get_frame(session_guid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return GameFrameResponse(
        grid=frame.grid,
        width=frame.width,
        height=frame.height,
        state=frame.state,
        score=frame.score,
        level=frame.level,
        total_actions=frame.total_actions,
        available_actions=frame.available_actions,
        metadata={"session_guid": session_guid},
    )


# ──── End Session ────
@router.post("/end/{session_guid}")
def end_session(
    session_guid: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    engine: GameEngineService = Depends(get_engine),
):
    """End a play session and cleanup."""
    play_session = (
        db.query(PlaySession)
        .filter(PlaySession.session_guid == session_guid)
        .first()
    )
    if play_session and play_session.state == "NOT_FINISHED":
        play_session.state = "GAME_OVER"
        # Capture timing from engine before destroying
        try:
            frame = engine.get_frame(session_guid)
            meta = frame.metadata or {}
            if meta.get("total_time"):
                play_session.total_time = meta["total_time"]
            if meta.get("completed_levels"):
                play_session.level_stats = meta["completed_levels"]
        except Exception:
            pass
        AnalyticsService.record_play_end(db, play_session)

    try:
        engine.destroy_instance(session_guid)
    except Exception:
        pass

    return {"detail": "Session ended"}


# ──── Get Color Palette ────
@router.get("/palette")
def get_palette():
    """Return the ARC-AGI color palette."""
    return GameEngineService.get_color_palette()
