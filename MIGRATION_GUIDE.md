# Fine-Tuning Studio Migration Guide

## Overview

The Fine-Tuning Studio has been restructured and integrated into the main `fine_tuning` module to simplify the codebase and improve maintainability. This guide helps you migrate from the old `fine_tuning_studio` structure to the new consolidated structure.

## What Changed

### Directory Structure Changes

**Before (Old Structure):**
```
src/use_cases/fine_tuning_studio/          # 7+ levels deep
├── backend/
│   ├── api/main.py                        # Level 6
│   ├── auth/auth_handler.py               # Level 6
│   ├── collaboration/collaboration_manager.py  # Level 6
│   ├── deployment/deployer.py             # Level 6
│   └── websocket/server.py                # Level 6
└── frontend/
    ├── app/dashboard/page.tsx             # Level 7
    └── app/experiments/new/page.tsx       # Level 8 ❌
```

**After (New Structure):**
```
src/use_cases/fine_tuning/                 # Consolidated
├── api/                                   # Level 4 ✅
│   ├── main.py                           # Core FastAPI app
│   ├── experiments.py                    # Experiment endpoints
│   ├── datasets.py                       # Dataset endpoints
│   ├── deployments.py                    # Deployment endpoints
│   ├── auth.py                           # Authentication
│   ├── collaboration.py                  # Team features
│   ├── websocket.py                      # Real-time updates
│   └── models.py                         # Pydantic models
├── web/                                   # Level 4 ✅
│   ├── components/                       # React components
│   ├── pages/                            # Page routes (flattened)
│   └── hooks/                            # React hooks
└── deployment/                            # Level 4 ✅
    └── deploy.py                         # Deployment logic
```

## Import Changes

### Python API Imports

**Old imports:**
```python
from src.use_cases.fine_tuning_studio.backend.api.main import app
from src.use_cases.fine_tuning_studio.backend.auth.auth_handler import AuthHandler
from src.use_cases.fine_tuning_studio.backend.collaboration.collaboration_manager import CollaborationManager
from src.use_cases.fine_tuning_studio.backend.deployment.deployer import DeploymentPipeline
```

**New imports:**
```python
from src.use_cases.fine_tuning.api import api_app
from src.use_cases.fine_tuning.api import AuthHandler
from src.use_cases.fine_tuning.api import CollaborationManager
from src.use_cases.fine_tuning.deployment import DeploymentPipeline
```

### Simplified imports using the main module:
```python
# Recommended approach - import from main fine_tuning module
from src.use_cases.fine_tuning import (
    api_app,
    AuthHandler,
    CollaborationManager,
    DeploymentPipeline,
    ExperimentCreate,
    Experiment,
    Dataset,
    Deployment
)
```

### Frontend/React Imports

**Old imports:**
```typescript
import Navigation from '../../components/Navigation';
import { useAuth } from '../../../hooks/useAuth';
import ExperimentCard from '../../../components/experiments/ExperimentCard';
```

**New imports:**
```typescript
import Navigation from '../components/Navigation';
import { useAuth } from '../hooks/useAuth';
import ExperimentCard from '../components/ExperimentCard';
```

## API Changes

### Endpoints Structure

The API has been reorganized into focused modules:

**Experiments API** (`/api/experiments/`)
- `POST /api/experiments/` - Create experiment
- `GET /api/experiments/` - List experiments
- `GET /api/experiments/{id}` - Get experiment details
- `POST /api/experiments/{id}/start` - Start experiment
- `POST /api/experiments/{id}/stop` - Stop experiment

**Datasets API** (`/api/datasets/`)
- `POST /api/datasets/` - Create dataset
- `POST /api/datasets/upload` - Upload dataset file
- `GET /api/datasets/` - List datasets
- `GET /api/datasets/{id}/preview` - Preview dataset

**Deployments API** (`/api/deployments/`)
- `POST /api/deployments/` - Create deployment
- `GET /api/deployments/` - List deployments
- `GET /api/deployments/{id}/status` - Get deployment status

**A/B Testing API** (`/api/ab-testing/`)
- `POST /api/ab-testing/` - Create A/B test
- `GET /api/ab-testing/` - List A/B tests
- `POST /api/ab-testing/{id}/compare` - Compare models

### Model Classes

All Pydantic models are now in `api/models.py`:

```python
from src.use_cases.fine_tuning.api.models import (
    # Configuration
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    Recipe,

    # Experiments
    ExperimentCreate,
    Experiment,
    ExperimentStatus,

    # Datasets
    Dataset,
    DatasetCreate,

    # Deployments
    Deployment,
    DeploymentConfig,

    # A/B Testing
    ABTest,
    ABTestConfig
)
```

## Migration Steps

### Step 1: Update Import Statements

Replace all imports from `fine_tuning_studio` with imports from `fine_tuning`:

```bash
# Find and replace in your codebase
find . -name "*.py" -exec sed -i 's/fine_tuning_studio/fine_tuning/g' {} +
```

### Step 2: Update API Endpoints

If you're calling the API directly, update endpoint paths:

**Old:**
```
POST /experiments
GET /datasets
POST /deploy
```

**New:**
```
POST /api/experiments/
GET /api/datasets/
POST /api/deployments/
```

### Step 3: Update Frontend Routes

Update React/Next.js imports and component references:

```bash
# Update TypeScript imports
find . -name "*.tsx" -exec sed -i 's|../../components/|../components/|g' {} +
find . -name "*.tsx" -exec sed -i 's|../../../hooks/|../hooks/|g' {} +
```

### Step 4: Update Configuration

If you have configuration files pointing to the old structure, update them:

```json
{
  "apiEndpoint": "/api",
  "webRoot": "/web",
  "deploymentService": "src.use_cases.fine_tuning.deployment"
}
```

## New Features Available

The restructured module now provides:

### 1. Clean Public API
```python
# Everything you need in one import
from src.use_cases.fine_tuning import *
```

### 2. Focused API Modules
- **Experiments**: Full experiment lifecycle management
- **Datasets**: Dataset upload, validation, and management
- **Deployments**: Multi-provider model deployment
- **A/B Testing**: Model comparison and testing

### 3. Improved Web Interface
- Flattened component structure
- Simplified page routing
- Better TypeScript organization

### 4. Enhanced Deployment
- Multiple provider support (HuggingFace, local, cloud)
- Template-based configuration
- Health monitoring and scaling

## Breaking Changes

### 1. Removed Deep Nesting
- Maximum 4 directory levels enforced
- Simplified import paths
- Better module organization

### 2. Consolidated Modules
- `collaboration_manager.py` → `api/collaboration.py`
- `auth_handler.py` → `api/auth.py`
- `deployer.py` → `deployment/deploy.py`

### 3. Restructured Frontend
- App directory flattened to pages directory
- Components consolidated in single directory
- Simplified routing structure

## Compatibility Notes

### Backward Compatibility
- Old `fine_tuning_studio` imports will fail
- API endpoint paths have changed
- Frontend component paths have changed

### Dependencies
The new structure maintains the same external dependencies:
- FastAPI for API server
- React/Next.js for frontend
- Pydantic for data models
- Optional: asyncio, uvicorn, etc.

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Error: ModuleNotFoundError: No module named 'fine_tuning_studio'
# Fix: Update imports to use 'fine_tuning'
from src.use_cases.fine_tuning.api import models
```

**2. API Endpoint Not Found**
```bash
# Error: 404 on /experiments
# Fix: Use new endpoint paths
curl -X POST /api/experiments/
```

**3. Frontend Build Errors**
```typescript
// Error: Cannot resolve '../../../components/Navigation'
// Fix: Use flattened paths
import Navigation from '../components/Navigation';
```

### Getting Help

1. Check the new directory structure in `src/use_cases/fine_tuning/`
2. Review the API documentation at `/docs` endpoint
3. Test imports using the verification script: `python simple_verification.py`

## Benefits of New Structure

### 1. Improved Maintainability
- **Shallow hierarchy**: Maximum 4 levels vs. previous 7+ levels
- **Clear boundaries**: Separate API, web, and deployment concerns
- **Focused modules**: Single responsibility per module

### 2. Better Developer Experience
- **Shorter import paths**: Reduced complexity in imports
- **Logical grouping**: Related functionality grouped together
- **Clean APIs**: Well-defined public interfaces

### 3. Enhanced Scalability
- **Modular architecture**: Easy to extend individual components
- **Separation of concerns**: Frontend, API, and deployment isolated
- **Standard patterns**: Follows Python and React best practices

## Next Steps

1. **Update your code** using the migration steps above
2. **Test thoroughly** to ensure all functionality works
3. **Review new features** to take advantage of improvements
4. **Update documentation** to reflect the new structure

The restructured Fine-Tuning Studio provides the same powerful functionality with a much cleaner, more maintainable architecture. Happy fine-tuning! 🚀
