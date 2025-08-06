"""
Authentication and authorization system for Fine-Tuning Studio
Implements JWT-based authentication with role-based access control
"""

import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Security
security = HTTPBearer()


class UserRole(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class Permission(str, Enum):
    CREATE_EXPERIMENT = "create_experiment"
    EDIT_EXPERIMENT = "edit_experiment"
    DELETE_EXPERIMENT = "delete_experiment"
    VIEW_EXPERIMENT = "view_experiment"
    DEPLOY_MODEL = "deploy_model"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_DATASETS = "manage_datasets"


# Role-Permission Mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.CREATE_EXPERIMENT,
        Permission.EDIT_EXPERIMENT,
        Permission.DELETE_EXPERIMENT,
        Permission.VIEW_EXPERIMENT,
        Permission.DEPLOY_MODEL,
        Permission.MANAGE_USERS,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_DATASETS,
    ],
    UserRole.DEVELOPER: [
        Permission.CREATE_EXPERIMENT,
        Permission.EDIT_EXPERIMENT,
        Permission.DELETE_EXPERIMENT,
        Permission.VIEW_EXPERIMENT,
        Permission.DEPLOY_MODEL,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_DATASETS,
    ],
    UserRole.VIEWER: [
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_ANALYTICS,
    ],
}


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole = UserRole.VIEWER


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[Permission] = []


class User(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool = True
    team_id: Optional[str] = None
    api_keys: List[str] = []
    settings: Dict[str, Any] = {}


# In-memory user storage (replace with database)
users_db: Dict[str, User] = {}
refresh_tokens_db: Dict[str, Dict] = {}
api_keys_db: Dict[str, str] = {}  # api_key -> user_id mapping


class AuthHandler:
    """Handles authentication and authorization logic"""

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, data: dict) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def decode_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            role: str = payload.get("role")

            if username is None or user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
                )

            token_data = TokenData(
                username=username,
                user_id=user_id,
                role=UserRole(role) if role else None,
                permissions=ROLE_PERMISSIONS.get(UserRole(role), []) if role else [],
            )
            return token_data

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return users_db.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in users_db.values():
            if user.username == username:
                return user
        return None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        import uuid

        # Check if username already exists
        if self.get_user_by_username(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered"
            )

        user = User(
            id=str(uuid.uuid4()),
            username=user_data.username,
            email=user_data.email,
            hashed_password=self.get_password_hash(user_data.password),
            role=user_data.role,
            created_at=datetime.utcnow(),
            last_login=None,
            is_active=True,
        )

        users_db[user.id] = user
        return user

    def generate_api_key(self, user_id: str) -> str:
        """Generate an API key for a user"""
        import secrets

        api_key = f"sk-{secrets.token_urlsafe(32)}"
        api_keys_db[api_key] = user_id

        if user_id in users_db:
            users_db[user_id].api_keys.append(api_key)

        return api_key

    def validate_api_key(self, api_key: str) -> Optional[User]:
        """Validate an API key and return the associated user"""
        user_id = api_keys_db.get(api_key)
        if user_id:
            return self.get_user(user_id)
        return None

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if a user has a specific permission"""
        user_permissions = ROLE_PERMISSIONS.get(user.role, [])
        return permission in user_permissions

    def update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        if user_id in users_db:
            users_db[user_id].last_login = datetime.utcnow()


# Create auth handler instance
auth_handler = AuthHandler()


# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """Get the current authenticated user from JWT token"""
    token = credentials.credentials
    token_data = auth_handler.decode_token(token)

    user = auth_handler.get_user(token_data.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[User]:
    """Get the current user if authenticated, otherwise return None"""
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_permission(permission: Permission):
    """Decorator to require specific permission for an endpoint"""

    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not auth_handler.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value}",
            )
        return current_user

    return permission_checker


def require_role(role: UserRole):
    """Decorator to require specific role for an endpoint"""

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Role required: {role.value}"
            )
        return current_user

    return role_checker


# API Key authentication
async def get_user_from_api_key(api_key: str = Depends(oauth2_scheme)) -> User:
    """Authenticate user using API key"""
    if api_key.startswith("sk-"):
        user = auth_handler.validate_api_key(api_key)
        if user:
            return user

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# Auth endpoints for FastAPI
def setup_auth_routes(app):
    """Add authentication routes to the FastAPI app"""

    @app.post("/api/auth/register", response_model=UserResponse)
    async def register(user_data: UserCreate):
        """Register a new user"""
        user = auth_handler.create_user(user_data)
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active,
        )

    @app.post("/api/auth/login", response_model=TokenResponse)
    async def login(user_data: UserLogin):
        """Login and receive access/refresh tokens"""
        user = auth_handler.authenticate_user(user_data.username, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
            )

        # Update last login
        auth_handler.update_last_login(user.id)

        # Create tokens
        access_token = auth_handler.create_access_token(
            data={"sub": user.username, "user_id": user.id, "role": user.role.value}
        )
        refresh_token = auth_handler.create_refresh_token(
            data={"sub": user.username, "user_id": user.id}
        )

        # Store refresh token
        refresh_tokens_db[refresh_token] = {"user_id": user.id, "created_at": datetime.utcnow()}

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    @app.post("/api/auth/refresh", response_model=TokenResponse)
    async def refresh_token(refresh_token: str):
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])

            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
                )

            user_id = payload.get("user_id")
            user = auth_handler.get_user(user_id)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
                )

            # Create new access token
            access_token = auth_handler.create_access_token(
                data={"sub": user.username, "user_id": user.id, "role": user.role.value}
            )

            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

    @app.post("/api/auth/logout")
    async def logout(refresh_token: str, current_user: User = Depends(get_current_user)):
        """Logout and invalidate refresh token"""
        if refresh_token in refresh_tokens_db:
            del refresh_tokens_db[refresh_token]

        return {"message": "Logged out successfully"}

    @app.get("/api/auth/me", response_model=UserResponse)
    async def get_me(current_user: User = Depends(get_current_user)):
        """Get current user information"""
        return UserResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            role=current_user.role,
            created_at=current_user.created_at,
            last_login=current_user.last_login,
            is_active=current_user.is_active,
        )

    @app.post("/api/auth/api-key")
    async def generate_api_key(current_user: User = Depends(get_current_user)):
        """Generate a new API key for the current user"""
        api_key = auth_handler.generate_api_key(current_user.id)
        return {"api_key": api_key}

    @app.get("/api/auth/permissions")
    async def get_permissions(current_user: User = Depends(get_current_user)):
        """Get current user's permissions"""
        permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        return {"role": current_user.role, "permissions": [p.value for p in permissions]}


# Initialize with default admin user (for development)
def init_default_users():
    """Create default users for development"""
    try:
        admin_user = UserCreate(
            username="admin", email="admin@example.com", password="admin123", role=UserRole.ADMIN
        )
        auth_handler.create_user(admin_user)
        logger.info("Default admin user created")
    except:
        pass  # User already exists


# Initialize default users
init_default_users()
