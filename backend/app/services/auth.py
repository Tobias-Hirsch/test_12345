from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session, selectinload
import redis
import json
from ..models import database
from ..schemas.schemas import TokenData, UserCreate
from ..services.msad_ldap import LDAPService
from ..core.redis_client import get_redis_client
from ..core.config import settings # Global import

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
 
# Initialize LDAPService
ldap_service = LDAPService()
 
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_user_by_username(db: Session, username: str):
    return db.query(database.User).options(
        selectinload(database.User.roles)
    ).filter(database.User.username == username).first()
 
def get_user_by_email(db: Session, email: str):
    return db.query(database.User).options(
        selectinload(database.User.roles)
    ).filter(database.User.email == email).first()
 
def get_user_by_phone(db: Session, phone: str):
    return db.query(database.User).options(
        selectinload(database.User.roles)
    ).filter(database.User.phone == phone).first()

def create_user(db: Session, user_create: UserCreate, is_active: bool = False) -> database.User:
    """
    Creates a new user in the database.
    If this is the first user, they are automatically assigned the 'admin' role.
    """
    # Check if this is the first user being created.
    user_count = db.query(database.User).count()

    hashed_password = get_password_hash(user_create.password)
    db_user = database.User(
        username=user_create.username,
        email=user_create.email,
        phone=user_create.phone,
        department=user_create.department,
        first_name=user_create.first_name,
        surname=user_create.surname,
        hashed_password=hashed_password,
        is_active=is_active
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # If this was the first user, assign them the admin role.
    if user_count == 0:
        admin_role = db.query(database.Role).filter(database.Role.name == "admin").first()
        if admin_role:
            db_user.roles.append(admin_role)
            db.commit()
            db.refresh(db_user)
            print(f"User '{db_user.username}' is the first user, assigned 'admin' role.")

    return db_user

def authenticate_user(db: Session, username: str, password: str):
    MSAD_SSO_ENABLED = settings.MSAD_SSO_ENABLED # Access here

    if MSAD_SSO_ENABLED:
        ldap_auth_success = ldap_service.authenticate_user(username, password)
        if not ldap_auth_success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="LDAP authentication failed. Invalid credentials or user disabled in AD.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # LDAP Kommentar
        user_attrs = ldap_service.get_user_attributes(username)
        if not user_attrs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User found in AD but attributes could not be retrieved.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Kommentar
        account_status = ldap_service.parse_user_account_control(user_attrs.get(ldap_service.user_account_control_attr, 0))
        is_active_in_ad = account_status["is_active"]

        user = get_user_by_username(db, username)
        if user:
            # Kommentar
            user.email = user_attrs.get("email", user.email)
            user.department = user_attrs.get("department", user.department)
            user.security_level = user_attrs.get("security_level", user.security_level)
            user.is_active = is_active_in_ad # Hinweis
            # TODO: Kommentar
            # user.roles = ...
            # user.permissions = ...
        else:
            # Kommentar
            user = database.User(
                username=username,
                email=user_attrs.get("email"),
                department=user_attrs.get("department"),
                security_level=user_attrs.get("security_level"),
                is_active=is_active_in_ad,
                # Kommentar
                hashed_password=None,
                provider="msad_ldap", # Hinweis
                provider_id=username # Hinweis
            )
            db.add(user)
        
        db.commit()
        db.refresh(user)
        return user
    else:
        # Kommentar
        user = get_user_by_username(db, username)
        if not user:
            user = get_user_by_email(db, username)
            if not user:
                user = get_user_by_phone(db, username)

        if not user or user.hashed_password is None or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(database.get_db),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    print("--- JWT DECODING DEBUG START ---")
    try:
        # IMPORTANT: Do NOT log the full secret key in production. This is for temporary debugging.
        print(f"Attempting to decode token with SECRET_KEY starting with: {settings.SECRET_KEY[:4]}...")
        print(f"Using algorithm: {settings.ALGORITHM}")
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        username: str = payload.get("sub")
        print(f"Token successfully decoded. Payload: {payload}")
        print(f"Username from token ('sub'): {username}")

        if username is None:
            print("ERROR: 'sub' field is missing in token payload.")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        print(f"ERROR: JWTError occurred during decoding: {e}")
        raise credentials_exception
    except Exception as e:
        print(f"An unexpected error occurred during token decoding: {e}")
        raise credentials_exception

    # Try to get user from Redis cache
    user_cache_key = f"user:{token_data.username}"
    cached_user_data = redis_client.get(user_cache_key)

    if cached_user_data:
        # If user data is in cache, we still need to fetch the full user object
        # with its relationships from the database to ensure all parts of the app
        # that rely on user.roles or user.permissions work correctly.
        user = db.query(database.User).options(
            selectinload(database.User.roles)
        ).filter(database.User.username == token_data.username).first()
        
        if user:
            return user
        else:
            # If user is not found in DB despite being in cache (edge case),
            # invalidate cache and raise exception.
            redis_client.delete(user_cache_key)
            raise credentials_exception

    # If not in cache, fetch from DB
    print(f"Querying database for username: '{token_data.username}'")
    user = db.query(database.User).options(
        selectinload(database.User.roles)
    ).filter(database.User.username == token_data.username).first()

    if user is None:
        print(f"ERROR: User '{token_data.username}' not found in the database.")
        raise credentials_exception
    
    print(f"SUCCESS: User '{user.username}' found in database. ID: {user.id}, Active: {user.is_active}")
    print("--- JWT DECODING DEBUG END ---")

    # Cache user data in Redis
    user_data_to_cache = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "phone": user.phone,
        "department": user.department,
        "security_level": user.security_level,
        "is_active": user.is_active,
        "avatar": user.avatar,
        "provider": user.provider,
        "provider_id": user.provider_id,
        # Do not cache hashed_password or sensitive tokens
        # Cache roles and permissions
        "roles": [role.name for role in user.roles],
        "permissions": [] # Kept for frontend compatibility, but now empty.
    }
    redis_client.setex(user_cache_key, settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, json.dumps(user_data_to_cache))

    return user

async def get_current_active_user(current_user: database.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

# TODO: Add functions for password reset token generation and verification