"""
FastAPI backend for Fine-Tuning Studio - Main API server
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ab_testing import router as ab_testing_router
from .datasets import router as datasets_router
from .deployments import router as deployments_router

# Import routers from sub-modules
from .experiments import router as experiments_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Fine-Tuning Studio API",
    description="Backend API for managing fine-tuning experiments",
    version="1.0.0",
)

# Configure CORS with configuration system
try:
    from ...config.settings import get_settings

    settings = get_settings()
    cors_origins = settings.server.cors_origins
    cors_allow_credentials = settings.server.cors_allow_credentials
except ImportError:
    # Fallback for backward compatibility
    cors_origins = ["http://localhost:3001", "http://localhost:3000"]
    cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(experiments_router, prefix="/api/experiments", tags=["experiments"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
app.include_router(deployments_router, prefix="/api/deployments", tags=["deployments"])
app.include_router(ab_testing_router, prefix="/api/ab-testing", tags=["ab-testing"])


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "fine-tuning-api"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fine-Tuning Studio API", "version": "1.0.0", "docs": "/docs"}


# Add authentication and websocket setup
try:
    from .auth import setup_auth_routes

    setup_auth_routes(app)
    logger.info("Authentication routes added")
except ImportError:
    logger.warning("Authentication module not available")

try:
    from .websocket import setup_websocket_routes

    setup_websocket_routes(app)
    logger.info("WebSocket routes added")
except ImportError:
    logger.warning("WebSocket module not available")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
