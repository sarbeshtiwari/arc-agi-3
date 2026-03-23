from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import uuid
import shutil
from datetime import datetime

from app.auth import get_admin_user, get_current_user
from app.config import settings
from app.database import get_db
from app.models import Game, GameAnalytics, PlaySession, User
from app.schemas import GameResponse, GameToggleRequest, GameUpdate
from app.services.game_manager import GameManagerService, GameValidationError
from app.services.game_engine import GameEngineService

router = APIRouter(prefix="/api/games", tags=["games"])

# Shared engine service for previews
_preview_engine: GameEngineService | None = None


def get_game_manager() -> GameManagerService:
    return GameManagerService(settings.ENVIRONMENT_FILES_DIR)


def _get_preview_engine() -> GameEngineService:
    global _preview_engine
    if _preview_engine is None:
        _preview_engine = GameEngineService(settings.ENVIRONMENT_FILES_DIR)
    return _preview_engine


# ──── Public: List Active Games (no auth) ────
@router.get("/public", response_model=list[GameResponse])
def list_public_games(db: Session = Depends(get_db)):
    """List all active games. No authentication required."""
    return (
        db.query(Game)
        .filter(Game.is_active == True)
        .order_by(Game.created_at.desc())
        .all()
    )


# ──── Public: Homepage analytics (no auth) ────
@router.get("/public/stats")
def get_public_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    active_games = db.query(Game).filter(Game.is_active == True).all()

    total_plays = sum(g.total_plays or 0 for g in active_games)
    total_wins = sum(g.total_wins or 0 for g in active_games)
    win_rate = round((total_wins / total_plays * 100), 1) if total_plays > 0 else 0

    top_played = sorted(active_games, key=lambda g: g.total_plays or 0, reverse=True)[:3]

    top_performers = {}
    for g in active_games:
        fastest = (
            db.query(PlaySession)
            .filter(
                PlaySession.game_id == g.id,
                PlaySession.state == "WIN",
                PlaySession.total_time > 0,
            )
            .order_by(PlaySession.total_time.asc())
            .first()
        )
        if fastest:
            from app.models import User as UserModel
            user = db.query(UserModel).filter(UserModel.id == fastest.user_id).first() if fastest.user_id else None
            top_performers[g.game_id] = {
                "player": fastest.player_name or (user.username if user else "Anonymous"),
                "time": round(fastest.total_time or 0, 2),
                "actions": fastest.total_actions or 0,
            }

    game_stats = []
    for g in top_played:
        wins = g.total_wins or 0
        plays = g.total_plays or 0
        game_stats.append({
            "game_id": g.game_id,
            "name": g.name or g.game_id,
            "total_plays": plays,
            "total_wins": wins,
            "win_rate": round((wins / plays * 100), 1) if plays > 0 else 0,
            "top_performer": top_performers.get(g.game_id),
            "levels": len(g.baseline_actions) if g.baseline_actions else 0,
        })

    featured = game_stats[0] if game_stats else None

    return {
        "total_plays": total_plays,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "total_games": len(active_games),
        "featured_game": featured,
        "top_games": game_stats,
    }


# ──── Public: Get Single Game (no auth) ────
@router.get("/public/{game_id}", response_model=GameResponse)
def get_public_game(game_id: str, db: Session = Depends(get_db)):
    """Get a single active game. No authentication required."""
    game = db.query(Game).filter(Game.game_id == game_id, Game.is_active == True).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


# ──── Public: Recent plays / leaderboard for a game (no auth) ────
@router.get("/public/{game_id}/plays")
def get_public_game_plays(
    game_id: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get recent plays for a game (public, no auth)."""
    game = db.query(Game).filter(Game.game_id == game_id, Game.is_active == True).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    sessions = (
        db.query(PlaySession)
        .filter(PlaySession.game_id == game.id, PlaySession.state.in_(["WIN", "GAME_OVER"]))
        .order_by(PlaySession.ended_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "player_name": s.player_name or "Anonymous",
            "state": s.state,
            "score": s.score,
            "total_actions": s.total_actions,
            "total_time": s.total_time,
            "level": s.current_level,
            "level_stats": s.level_stats or [],
            "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        }
        for s in sessions
    ]


# ──── Public: Per-game analytics (no auth) ────
@router.get("/public/{game_id}/stats")
def get_public_game_stats(game_id: str, db: Session = Depends(get_db)):
    game = db.query(Game).filter(Game.game_id == game_id, Game.is_active == True).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    plays = game.total_plays or 0
    wins = game.total_wins or 0
    win_rate = round((wins / plays * 100), 1) if plays > 0 else 0

    # Fastest win
    fastest = (
        db.query(PlaySession)
        .filter(PlaySession.game_id == game.id, PlaySession.state == "WIN", PlaySession.total_time > 0)
        .order_by(PlaySession.total_time.asc())
        .first()
    )
    top_performer = None
    if fastest:
        from app.models import User as UserModel
        user = db.query(UserModel).filter(UserModel.id == fastest.user_id).first() if fastest.user_id else None
        top_performer = {
            "player": fastest.player_name or (user.username if user else "Anonymous"),
            "time": round(fastest.total_time or 0, 2),
            "actions": fastest.total_actions or 0,
        }

    # Average completion time (wins only)
    win_sessions = (
        db.query(PlaySession)
        .filter(PlaySession.game_id == game.id, PlaySession.state == "WIN", PlaySession.total_time > 0)
        .all()
    )
    avg_time = round(sum(s.total_time for s in win_sessions) / len(win_sessions), 2) if win_sessions else 0

    # Recent players
    recent = (
        db.query(PlaySession)
        .filter(PlaySession.game_id == game.id)
        .order_by(PlaySession.started_at.desc())
        .limit(5)
        .all()
    )
    recent_players = []
    for s in recent:
        from app.models import User as UserModel
        user = db.query(UserModel).filter(UserModel.id == s.user_id).first() if s.user_id else None
        recent_players.append({
            "player": s.player_name or (user.username if user else "Anonymous"),
            "state": s.state,
            "time": round(s.total_time or 0, 2),
        })

    return {
        "game_id": game_id,
        "name": game.name,
        "total_plays": plays,
        "total_wins": wins,
        "win_rate": win_rate,
        "avg_completion_time": avg_time,
        "top_performer": top_performer,
        "recent_players": recent_players,
        "levels": len(game.baseline_actions) if game.baseline_actions else 0,
    }


# ──── Public: Get game preview (initial frame grid, no auth) ────
@router.get("/public/{game_id}/preview")
def get_game_preview(game_id: str, db: Session = Depends(get_db)):
    """Return the initial frame grid for a game (for thumbnail preview)."""
    game = db.query(Game).filter(Game.game_id == game_id, Game.is_active == True).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    engine = _get_preview_engine()
    session_id = f"preview_{uuid.uuid4().hex[:8]}"

    try:
        engine.create_instance(
            session_id=session_id,
            game_id=game.game_id,
            game_code=game.game_code,
            game_dir=game.local_dir,
            seed=0,
        )
        frame = engine.get_frame(session_id)
        grid = frame.grid
        width = frame.width
        height = frame.height
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {e}")
    finally:
        try:
            engine.destroy_instance(session_id)
        except Exception:
            pass

    return {
        "game_id": game_id,
        "grid": grid,
        "width": width,
        "height": height,
    }


# ──── List Games (auth required) ────
@router.get("/", response_model=list[GameResponse])
def list_games(
    active_only: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = db.query(Game)
    if active_only:
        query = query.filter(Game.is_active == True)
    return query.order_by(Game.created_at.desc()).all()


# ──── Get Single Game ────
@router.get("/{game_id}", response_model=GameResponse)
def get_game(
    game_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


# ──── Upload Game ────
@router.post("/upload", response_model=GameResponse)
async def upload_game(
    game_file: UploadFile = File(..., description="The game .py file"),
    metadata_file: UploadFile = File(..., description="The metadata.json file"),
    name: str = Form("", description="Human-readable game name (defaults to game_id)"),
    description: str = Form("", description="Game description"),
    game_rules: str = Form("", description="Game rules / how to play"),
    game_owner_name: str = Form("", description="Game owner/creator name"),
    game_drive_link: str = Form("", description="Google Drive or download link"),
    game_video_link: str = Form("", description="Video demo link"),
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    manager: GameManagerService = Depends(get_game_manager),
):
    """Upload a new game (game.py + metadata.json)."""
    game_py_content = await game_file.read()
    metadata_content = await metadata_file.read()

    try:
        result = manager.upload_game(game_py_content, metadata_content)
    except GameValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    existing = db.query(Game).filter(Game.game_id == result["game_id"]).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Game '{result['game_id']}' already exists. Delete it first or use a different version.",
        )

    # Create DB record
    metadata = result["metadata"]
    game_name = name if name else result["game_id"]
    game = Game(
        game_id=result["game_id"],
        name=game_name,
        description=description,
        game_rules=game_rules,
        game_owner_name=game_owner_name or None,
        game_drive_link=game_drive_link or None,
        game_video_link=game_video_link or None,
        version=result["version"],
        game_code=result["game_code"],
        is_active=False,
        default_fps=metadata.get("default_fps", 5),
        baseline_actions=metadata.get("baseline_actions"),
        tags=metadata.get("tags"),
        game_file_path=result["game_file_path"],
        metadata_file_path=result["metadata_file_path"],
        local_dir=result["local_dir"],
        uploaded_by=admin.id,
    )
    db.add(game)
    db.commit()
    db.refresh(game)
    return game


# ──── Update Game ────
@router.put("/{game_id}", response_model=GameResponse)
def update_game(
    game_id: str,
    request: GameUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    if request.name is not None:
        game.name = request.name
    if request.description is not None:
        game.description = request.description
    if request.game_rules is not None:
        game.game_rules = request.game_rules
    if request.game_owner_name is not None:
        game.game_owner_name = request.game_owner_name
    if request.game_drive_link is not None:
        game.game_drive_link = request.game_drive_link
    if request.game_video_link is not None:
        game.game_video_link = request.game_video_link
    if request.is_active is not None:
        game.is_active = request.is_active
    if request.default_fps is not None:
        game.default_fps = request.default_fps
    if request.tags is not None:
        game.tags = request.tags

    db.commit()
    db.refresh(game)
    return game


# ──── Update Game Files ────
@router.put("/{game_id}/files", response_model=GameResponse)
async def update_game_files(
    game_id: str,
    game_file: UploadFile = File(None, description="Updated game .py file"),
    metadata_file: UploadFile = File(None, description="Updated metadata.json file"),
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    manager: GameManagerService = Depends(get_game_manager),
):
    """Update game files (game.py and/or metadata.json) for an existing game."""
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    if not game_file and not metadata_file:
        raise HTTPException(status_code=400, detail="No files provided")

    game_dir = game.local_dir
    if not game_dir or not os.path.exists(game_dir):
        raise HTTPException(status_code=500, detail="Game directory not found on disk")

    # Update game.py if provided
    if game_file:
        game_py_bytes = await game_file.read()
        try:
            game_py_str = game_py_bytes.decode("utf-8")
            manager.validate_game_file(game_py_str, game.game_code)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid game file: {e}")

        game_py_path = os.path.join(game_dir, f"{game.game_code}.py")
        with open(game_py_path, "wb") as f:
            f.write(game_py_bytes)
        game.game_file_path = game_py_path

    # Update metadata.json if provided
    if metadata_file:
        metadata_bytes = await metadata_file.read()
        try:
            metadata_str = metadata_bytes.decode("utf-8")
            metadata = manager.validate_metadata(metadata_str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata file: {e}")

        metadata_path = os.path.join(game_dir, "metadata.json")
        with open(metadata_path, "wb") as f:
            f.write(metadata_bytes)
        game.metadata_file_path = metadata_path

        # Update game fields from new metadata
        if metadata.get("default_fps"):
            game.default_fps = metadata["default_fps"]
        if metadata.get("baseline_actions") is not None:
            game.baseline_actions = metadata["baseline_actions"]
        if metadata.get("tags") is not None:
            game.tags = metadata["tags"]

    db.commit()
    db.refresh(game)
    return game


# ──── Toggle Active/Inactive ────
@router.patch("/{game_id}/toggle", response_model=GameResponse)
def toggle_game(
    game_id: str,
    request: GameToggleRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    game.is_active = request.is_active
    db.commit()
    db.refresh(game)
    return game


# ──── Delete Game ────
@router.delete("/{game_id}")
def delete_game(
    game_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    manager: GameManagerService = Depends(get_game_manager),
):
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    manager.delete_game_files(game.game_code, game.version)

    db.query(PlaySession).filter(PlaySession.game_id == game.id).delete()
    db.query(GameAnalytics).filter(GameAnalytics.game_id == game.id).delete()

    db.delete(game)
    db.commit()
    return {"detail": f"Game '{game_id}' deleted"}


# ──── Clear Sessions & Analytics for a Game ────
@router.delete("/{game_id}/sessions")
def clear_game_sessions(
    game_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """Delete all play sessions and analytics for a specific game."""
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    session_count = db.query(PlaySession).filter(PlaySession.game_id == game.id).count()
    analytics_count = db.query(GameAnalytics).filter(GameAnalytics.game_id == game.id).count()

    db.query(PlaySession).filter(PlaySession.game_id == game.id).delete()
    db.query(GameAnalytics).filter(GameAnalytics.game_id == game.id).delete()

    game.total_plays = 0
    game.total_wins = 0
    game.avg_score = 0

    db.commit()
    return {"detail": f"Cleared {session_count} sessions and {analytics_count} analytics records"}


# ──── Get Game Source Code ────
@router.get("/{game_id}/source")
def get_game_source(
    game_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    manager: GameManagerService = Depends(get_game_manager),
):
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    content = manager.get_game_file_content(game.game_code, game.version)
    metadata = manager.get_metadata_content(game.game_code, game.version)

    return {
        "game_id": game.game_id,
        "source_code": content,
        "metadata": metadata,
    }


# ──── Sync Local Games ────
@router.post("/sync-local")
def sync_local_games(
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    manager: GameManagerService = Depends(get_game_manager),
):
    """Scan environment_files directory and register any unregistered games."""
    local_games = manager.list_local_games()
    added = []

    for lg in local_games:
        existing = db.query(Game).filter(Game.game_id == lg["game_id"]).first()
        if not existing:
            game = Game(
                game_id=lg["game_id"],
                name=lg["game_id"],
                version=lg["version"],
                game_code=lg["game_code"],
                game_file_path=str(
                    (
                        __import__("pathlib").Path(lg["local_dir"])
                        / f"{lg['game_code']}.py"
                    )
                ),
                metadata_file_path=str(
                    __import__("pathlib").Path(lg["local_dir"]) / "metadata.json"
                ),
                local_dir=lg["local_dir"],
                default_fps=lg["metadata"].get("default_fps", 5),
                baseline_actions=lg["metadata"].get("baseline_actions"),
                tags=lg["metadata"].get("tags"),
                uploaded_by=admin.id,
            )
            db.add(game)
            added.append(lg["game_id"])

    db.commit()
    return {"synced": len(added), "games": added}


# ──── Video Recording Endpoints ────

@router.post("/public/{game_id}/video")
async def upload_game_video(
    game_id: str,
    video: UploadFile = File(...),
    player_name: str = Form(""),
    db: Session = Depends(get_db),
):
    """Public: upload a gameplay recording for a live game."""
    game = db.query(Game).filter(Game.game_id == game_id, Game.is_active == True).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found or inactive")

    recordings_dir = os.path.join(game.local_dir, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in (player_name or "anonymous") if c.isalnum() or c in "_-")[:20]
    filename = f"{safe_name}_{timestamp}.webm"
    filepath = os.path.join(recordings_dir, filename)

    with open(filepath, "wb") as f:
        contents = await video.read()
        f.write(contents)

    size = os.path.getsize(filepath)
    return {"filename": filename, "size": size, "game_id": game_id}


@router.get("/{game_id}/videos")
def list_game_videos(
    game_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """Admin: list all recorded videos for a game."""
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    recordings_dir = os.path.join(game.local_dir, "recordings")
    if not os.path.exists(recordings_dir):
        return []

    videos = []
    for filename in sorted(os.listdir(recordings_dir), reverse=True):
        if not filename.endswith(".webm"):
            continue
        filepath = os.path.join(recordings_dir, filename)
        stat = os.stat(filepath)
        # Parse player name and timestamp from filename: player_YYYYMMDD_HHMMSS.webm
        parts = filename.rsplit("_", 2)
        player = parts[0] if len(parts) >= 3 else "unknown"
        videos.append({
            "filename": filename,
            "player": player,
            "size": stat.st_size,
            "created_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
            "url": f"/api/games/{game_id}/videos/{filename}",
        })

    return videos


@router.get("/{game_id}/videos/{filename}")
def serve_game_video(
    game_id: str,
    filename: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """Admin: serve a specific video file."""
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(game.local_dir, "recordings", safe_filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(filepath, media_type="video/webm", filename=safe_filename)


@router.delete("/{game_id}/videos/{filename}")
def delete_game_video(
    game_id: str,
    filename: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """Admin: delete a specific video."""
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    safe_filename = os.path.basename(filename)
    filepath = os.path.join(game.local_dir, "recordings", safe_filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Video not found")

    os.remove(filepath)
    return {"detail": f"Video '{safe_filename}' deleted"}
