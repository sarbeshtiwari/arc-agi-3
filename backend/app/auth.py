from datetime import datetime, timedelta
import logging

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


def require_page(page: str):
    """Dependency factory: check if the user has access to a specific page."""
    def checker(current_user: User = Depends(get_current_user)) -> User:
        # Admins always have full access
        if current_user.is_admin:
            return current_user
        allowed = current_user.allowed_pages or []
        if page not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You do not have access to '{page}'",
            )
        return current_user
    return checker


def create_default_admin(db: Session) -> None:
    all_pages = ["dashboard", "games", "upload", "requests", "users"]
    existing = db.query(User).filter(User.username == settings.DEFAULT_ADMIN_USERNAME).first()
    if not existing:
        admin = User(
            username=settings.DEFAULT_ADMIN_USERNAME,
            hashed_password=hash_password(settings.DEFAULT_ADMIN_PASSWORD),
            is_admin=True,
            is_active=True,
            allowed_pages=all_pages,
        )
        db.add(admin)
        db.commit()
        logger.info(f"[INIT] Default admin user created: {settings.DEFAULT_ADMIN_USERNAME}")
    elif not existing.allowed_pages:
        existing.allowed_pages = all_pages
        db.commit()
    if existing and not existing.is_admin:
        existing.is_admin = True
        existing.allowed_pages = all_pages
        db.commit()
        logger.info(f"[INIT] Fixed admin flag for: {settings.DEFAULT_ADMIN_USERNAME}")


def create_protected_super_admin(db: Session) -> None:
    """Ensure the protected super admin exists and is always admin + active."""
    all_pages = ["dashboard", "games", "upload", "requests", "users"]
    existing = db.query(User).filter(User.username == settings.PROTECTED_USERNAME).first()
    if not existing:
        user = User(
            username=settings.PROTECTED_USERNAME,
            hashed_password=hash_password(settings.PROTECTED_USERNAME),
            email=settings.PROTECTED_USERNAME,
            is_admin=True,
            is_active=True,
            allowed_pages=all_pages,
        )
        db.add(user)
        db.commit()
        logger.info(f"[INIT] Protected super admin created: {settings.PROTECTED_USERNAME}")
    else:
        changed = False
        if not existing.is_admin:
            existing.is_admin = True
            changed = True
        if not existing.is_active:
            existing.is_active = True
            changed = True
        if not existing.allowed_pages or set(existing.allowed_pages) != set(all_pages):
            existing.allowed_pages = all_pages
            changed = True
        if changed:
            db.commit()
            logger.info(f"[INIT] Protected super admin restored: {settings.PROTECTED_USERNAME}")
