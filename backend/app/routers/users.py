from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import get_admin_user, get_current_user, hash_password
from app.config import settings
from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, UserUpdate

router = APIRouter(prefix="/api/users", tags=["users"])


def _is_protected(user) -> bool:
    return user.username == settings.PROTECTED_USERNAME


@router.get("/", response_model=list[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """List all users (admin only)."""
    return db.query(User).order_by(User.created_at.desc()).all()


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    request: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if _is_protected(user):
        raise HTTPException(
            status_code=403,
            detail="This user is protected and cannot be modified",
        )

    if request.email is not None:
        user.email = request.email
    if request.is_admin is not None:
        user.is_admin = request.is_admin
    if request.is_active is not None:
        user.is_active = request.is_active
    if request.allowed_pages is not None:
        user.allowed_pages = request.allowed_pages
    if request.password is not None:
        user.hashed_password = hash_password(request.password)

    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}")
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if _is_protected(user):
        raise HTTPException(status_code=403, detail="This user is protected and cannot be deleted")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    db.delete(user)
    db.commit()
    return {"detail": "User deleted"}


# ──── Protected user: change password with secret code ────
from pydantic import BaseModel


class ProtectedPasswordChange(BaseModel):
    secret_code: str
    new_password: str


@router.post("/protected/change-password", response_model=UserResponse)
def change_protected_password(
    request: ProtectedPasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Change password for the protected super admin. Requires secret code.
    Only the protected user themselves can call this."""
    if current_user.username != settings.PROTECTED_USERNAME:
        raise HTTPException(status_code=403, detail="Only the protected user can change their own password")

    if request.secret_code != settings.PROTECTED_SECRET_CODE:
        raise HTTPException(status_code=403, detail="Invalid secret code")

    if len(request.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    current_user.hashed_password = hash_password(request.new_password)
    db.commit()
    db.refresh(current_user)
    return current_user
