from contextlib import asynccontextmanager
from datetime import datetime
import logging
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.auth import create_default_admin, create_protected_super_admin
from app.models import Game, TempGameSession, now_ist
from app.services.game_manager import GameManagerService
from app.routers import auth, games, player, analytics, users, requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arc-agi")


def sync_local_games_on_startup(db):
    """Scan environment_files and register any unregistered games in the DB.
    Only syncs games that are already in the DB but missing records (e.g. after DB reset).
    Does NOT auto-register new games -- those must go through upload or approval flow.
    Skips _requests directory.
    """
    manager = GameManagerService(settings.ENVIRONMENT_FILES_DIR)
    local_games = manager.list_local_games()
    synced = []

    for lg in local_games:
        # Skip the _requests directory
        if "/_requests/" in lg["local_dir"] or "\\_requests\\" in lg["local_dir"]:
            continue

        existing = db.query(Game).filter(Game.game_id == lg["game_id"]).first()
        if existing:
            # Sync metadata fields from disk to DB (in case files were updated)
            meta = lg.get("metadata", {})
            changed = False
            ba = meta.get("baseline_actions")
            if ba is not None and existing.baseline_actions != ba:
                existing.baseline_actions = ba
                changed = True
            fps = meta.get("default_fps")
            if fps is not None and existing.default_fps != fps:
                existing.default_fps = fps
                changed = True
            tags = meta.get("tags")
            if tags is not None and existing.tags != tags:
                existing.tags = tags
                changed = True
            if changed:
                synced.append(lg["game_id"])
            continue

    if synced:
        db.commit()
        logger.info(f"[STARTUP] Synced metadata for {len(synced)} game(s): {synced}")
    else:
        logger.info("[STARTUP] All game metadata up to date")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("[STARTUP] Database tables created/verified")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to connect to database: {e}", exc_info=True)
        raise

    db = SessionLocal()
    try:
        create_default_admin(db)
        create_protected_super_admin(db)
        sync_local_games_on_startup(db)
        expired = db.query(TempGameSession).filter(
            TempGameSession.expires_at < now_ist()
        ).all()
        if expired:
            for t in expired:
                db.delete(t)
            db.commit()
            logger.info(f"[STARTUP] Cleaned up {len(expired)} expired temp game sessions")
    except Exception as e:
        logger.error(f"[STARTUP] Startup task failed: {e}", exc_info=True)
        db.close()
        raise
    finally:
        db.close()

    logger.info(f"[STARTUP] {settings.APP_NAME} is running")
    logger.info(f"[STARTUP] Environment files dir: {settings.ENVIRONMENT_FILES_DIR}")
    yield
    logger.info("[SHUTDOWN] Cleaning up...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Admin panel for managing ARC-AGI-3 game environments",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 1)

        if response.status_code >= 500:
            logger.error(f"{method} {path} -> {response.status_code} ({duration}ms)")
        elif response.status_code >= 400:
            logger.warning(f"{method} {path} -> {response.status_code} ({duration}ms)")

        return response
    except Exception as e:
        duration = round((time.time() - start) * 1000, 1)
        logger.error(f"{method} {path} -> CRASH ({duration}ms)\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(games.router)
app.include_router(player.router)
app.include_router(analytics.router)
app.include_router(requests.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME}
