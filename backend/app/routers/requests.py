import json
import os
import ipaddress
import socket
from datetime import datetime
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth import get_admin_user
from app.config import settings
from app.database import get_db
from app.models import Game, GameRequest, User, now_ist
from app.schemas import GameRequestResponse, GameRequestReview, GameResponse
from app.services.game_manager import GameManagerService, GameValidationError

router = APIRouter(prefix="/api/requests", tags=["game-requests"])

REQUESTS_DIR = os.path.join(settings.ENVIRONMENT_FILES_DIR, "_requests")


def get_game_manager() -> GameManagerService:
    return GameManagerService(settings.ENVIRONMENT_FILES_DIR)


def _validate_url(url: str, field_name: str):
    """Validate that a URL is well-formed, not targeting private/internal IPs, and does not return 404."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail=f"{field_name}: URL must start with http:// or https://")

    # SSRF protection: block requests to private/internal IPs
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail=f"{field_name}: URL has no valid hostname")

    if hostname in ("localhost", "localhost.localdomain"):
        raise HTTPException(status_code=400, detail=f"{field_name}: URLs targeting localhost are not allowed")

    try:
        resolved_ips = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # DNS resolution failed — let it pass (don't block on transient DNS issues)
        resolved_ips = []

    for family, _type, _proto, _canonname, sockaddr in resolved_ips:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}: URLs targeting private/internal networks are not allowed",
            )

    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.head(url)
            if resp.status_code == 404:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field_name}: URL returned 404 (not found). Please check the link.",
                )
    except httpx.RequestError:
        # Network error / DNS failure / timeout -- let it pass, don't block upload
        pass


# ──── Public: Submit a game request ────
@router.post("/submit", response_model=GameRequestResponse)
async def submit_game_request(
    game_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...),
    requester_name: str = Form(...),
    requester_email: str = Form(""),
    message: str = Form(""),
    description: str = Form(""),
    game_rules: str = Form(""),
    game_owner_name: str = Form(""),
    game_drive_link: str = Form(""),
    game_video_link: str = Form(""),
    db: Session = Depends(get_db),
):
    """Public endpoint: submit a game for review. No auth required."""
    # Validate URLs before processing files
    drive_link = game_drive_link.strip() or None
    video_link = game_video_link.strip() or None
    if drive_link:
        _validate_url(drive_link, "Game drive link")
    if video_link:
        _validate_url(video_link, "Game video link")

    game_py_bytes = await game_file.read()
    metadata_bytes = await metadata_file.read()

    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata.json")

    game_id = metadata.get("game_id")
    if not game_id:
        raise HTTPException(status_code=400, detail='metadata.json must contain "game_id"')

    # Check uniqueness: game_id must not exist in games or pending requests
    existing_game = db.query(Game).filter(Game.game_id == game_id).first()
    if existing_game:
        raise HTTPException(status_code=400, detail=f'Game "{game_id}" already exists')

    existing_request = (
        db.query(GameRequest)
        .filter(GameRequest.game_id == game_id, GameRequest.status == "pending")
        .first()
    )
    if existing_request:
        raise HTTPException(
            status_code=400,
            detail=f'A request for "{game_id}" is already pending review',
        )

    if "-" in game_id:
        parts = game_id.rsplit("-", 1)
        game_code = parts[0]
        version = parts[1] if len(parts) > 1 else "v1"
    else:
        game_code = game_id
        version = "v1"

    req_dir = os.path.join(REQUESTS_DIR, game_id)
    os.makedirs(req_dir, exist_ok=True)

    game_py_path = os.path.join(req_dir, f"{game_code}.py")
    metadata_path = os.path.join(req_dir, "metadata.json")

    with open(game_py_path, "wb") as f:
        f.write(game_py_bytes)
    with open(metadata_path, "wb") as f:
        f.write(metadata_bytes)

    req = GameRequest(
        game_id=game_id,
        requester_name=requester_name.strip(),
        requester_email=requester_email.strip() or None,
        message=message.strip() or None,
        description=description.strip() or None,
        game_rules=game_rules.strip() or None,
        game_owner_name=game_owner_name.strip() or None,
        game_drive_link=drive_link,
        game_video_link=video_link,
        status="pending",
        game_file_path=game_py_path,
        metadata_file_path=metadata_path,
        local_dir=req_dir,
        game_file_content=game_py_bytes,
        metadata_file_content=metadata_bytes,
        game_code=game_code,
        version=version,
        tags=metadata.get("tags"),
        default_fps=metadata.get("default_fps", 5),
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req


# ──── Admin: List requests ────
@router.get("/", response_model=list[GameRequestResponse])
def list_requests(
    status: str = "pending",
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """List game requests filtered by status."""
    query = db.query(GameRequest)
    if status != "all":
        query = query.filter(GameRequest.status == status)
    return query.order_by(GameRequest.created_at.desc()).all()


# ──── Admin: Get request details ────
@router.get("/{request_id}", response_model=GameRequestResponse)
def get_request(
    request_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    req = db.query(GameRequest).filter(GameRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    return req


# ──── Admin: Get request source code ────
@router.get("/{request_id}/source")
def get_request_source(
    request_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    req = db.query(GameRequest).filter(GameRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    source_code = None
    metadata = None
    if req.game_file_content:
        source_code = req.game_file_content.decode("utf-8", errors="replace")
    elif req.game_file_path and os.path.exists(req.game_file_path):
        with open(req.game_file_path, "r") as f:
            source_code = f.read()
    if req.metadata_file_content:
        metadata = json.loads(req.metadata_file_content.decode("utf-8"))
    elif req.metadata_file_path and os.path.exists(req.metadata_file_path):
        with open(req.metadata_file_path, "r") as f:
            metadata = json.load(f)

    return {"source_code": source_code, "metadata": metadata}


# ──── Admin: Download request game files (for play testing) ────
@router.get("/{request_id}/files/{file_type}")
def get_request_file(
    request_id: str,
    file_type: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """Download a game file from a request. file_type: 'game' or 'metadata'."""
    from fastapi.responses import Response

    req = db.query(GameRequest).filter(GameRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    if file_type == "game":
        content = req.game_file_content
        if not content and req.game_file_path and os.path.exists(req.game_file_path):
            with open(req.game_file_path, "rb") as f:
                content = f.read()
        if not content:
            raise HTTPException(status_code=404, detail="Game file not found")
        return Response(content=content, media_type="text/x-python", headers={
            "Content-Disposition": f'attachment; filename="{req.game_id}.py"'
        })
    elif file_type == "metadata":
        content = req.metadata_file_content
        if not content and req.metadata_file_path and os.path.exists(req.metadata_file_path):
            with open(req.metadata_file_path, "rb") as f:
                content = f.read()
        if not content:
            raise HTTPException(status_code=404, detail="Metadata file not found")
        return Response(content=content, media_type="application/json", headers={
            "Content-Disposition": 'attachment; filename="metadata.json"'
        })
    else:
        raise HTTPException(status_code=400, detail="file_type must be 'game' or 'metadata'")


# ──── Admin: Approve or Reject ────
@router.post("/{request_id}/review")
def review_request(
    request_id: str,
    review: GameRequestReview,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    manager: GameManagerService = Depends(get_game_manager),
):
    req = db.query(GameRequest).filter(GameRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request already {req.status}")

    if review.action == "reject":
        req.status = "rejected"
        req.admin_note = review.admin_note
        req.reviewed_at = now_ist()
        req.reviewed_by = admin.id
        db.commit()
        db.refresh(req)
        return req

    if review.action != "approve":
        raise HTTPException(status_code=400, detail='action must be "approve" or "reject"')

    existing = db.query(Game).filter(Game.game_id == req.game_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f'Game "{req.game_id}" already exists')

    game_py_bytes = req.game_file_content
    metadata_bytes = req.metadata_file_content

    if not game_py_bytes or not metadata_bytes:
        try:
            with open(req.game_file_path, "rb") as f:
                game_py_bytes = f.read()
            with open(req.metadata_file_path, "rb") as f:
                metadata_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Game files not found. The request may need to be resubmitted: {e}")

    try:
        result = manager.upload_game(game_py_bytes, metadata_bytes)
    except GameValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    metadata = result["metadata"]
    game = Game(
        game_id=result["game_id"],
        name=result["game_id"],  # name = game_id
        description=req.description,
        game_rules=req.game_rules,
        game_owner_name=req.game_owner_name,
        game_drive_link=req.game_drive_link,
        game_video_link=req.game_video_link,
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

    db.delete(req)

    db.commit()

    import shutil
    if req.local_dir and os.path.exists(req.local_dir):
        shutil.rmtree(req.local_dir, ignore_errors=True)

    db.refresh(game)
    return GameResponse.model_validate(game)


# ──── Admin: Delete request ────
@router.delete("/{request_id}")
def delete_request(
    request_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    req = db.query(GameRequest).filter(GameRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    import shutil
    if req.local_dir and os.path.exists(req.local_dir):
        shutil.rmtree(req.local_dir, ignore_errors=True)

    db.delete(req)
    db.commit()
    return {"detail": "Request deleted"}
