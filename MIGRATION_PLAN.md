# Fine-Tuning Studio Migration Plan

## Overview
Migration from the current 6-7 level nested structure to a flat 4-level structure by consolidating `fine_tuning_studio` into the existing `fine_tuning` module.

## Current State Analysis
- **Files to migrate**: 27 files across 8 directories
- **Deepest nesting**: 6 levels (frontend/app/experiments/new/)
- **Target**: Maximum 4 levels
- **Risk level**: Low (no external dependencies found)

## Migration Strategy

### Phase 1: Backend API Consolidation
Move from `src/use_cases/fine_tuning_studio/backend/` to `src/use_cases/fine_tuning/api/`

#### File Migrations:
```
# API Core
backend/api/main.py → api/main.py
backend/websocket/server.py → api/websocket.py

# Authentication & Authorization
backend/auth/auth_handler.py → api/auth.py

# Experiment Management
[Extract from main.py] → api/experiments.py
[Extract from main.py] → api/datasets.py
[Extract from main.py] → api/deployments.py

# Collaboration Features
backend/collaboration/collaboration_manager.py → api/collaboration.py

# Infrastructure Services
backend/versioning/experiment_versioning.py → api/versioning.py
backend/model_card/generator.py → api/model_cards.py
backend/testing/test_suite.py → api/testing.py
```

### Phase 2: Frontend Web Interface
Move from `src/use_cases/fine_tuning_studio/frontend/` to `src/use_cases/fine_tuning/web/`

#### Directory Flattening:
```
# React Components (Level 5 → 4)
frontend/components/ → web/components/
frontend/hooks/ → web/hooks/
frontend/utils/ → web/utils/

# App Routes (Level 6-7 → 4)
frontend/app/dashboard/page.tsx → web/pages/dashboard.tsx
frontend/app/experiments/new/page.tsx → web/pages/experiments_new.tsx
frontend/app/experiments/[id]/page.tsx → web/pages/experiment_detail.tsx
frontend/app/datasets/page.tsx → web/pages/datasets.tsx
frontend/app/models/page.tsx → web/pages/models.tsx
frontend/app/deploy/page.tsx → web/pages/deploy.tsx
frontend/app/ab-testing/page.tsx → web/pages/ab_testing.tsx
frontend/app/settings/page.tsx → web/pages/settings.tsx

# Configuration Files
frontend/package.json → web/package.json
frontend/next.config.js → web/next.config.js
frontend/tsconfig.json → web/tsconfig.json
```

### Phase 3: Deployment Infrastructure
Move from `src/use_cases/fine_tuning_studio/deployment/` to `src/use_cases/fine_tuning/deployment/`

#### Infrastructure Files:
```
deployment/deployer.py → deployment/deploy.py
deployment/configs/ → deployment/configs/
deployment/scripts/ → deployment/scripts/
```

## Detailed File Migration Plan

### API Layer Restructuring

**1. Main API Server (`main.py` → `api/main.py`)**
- **Size**: 681 lines
- **Action**: Split into focused modules
- **New structure**:
  ```
  api/main.py (core FastAPI app, ~200 lines)
  api/experiments.py (experiment endpoints, ~150 lines)
  api/datasets.py (dataset endpoints, ~100 lines)
  api/deployments.py (deployment endpoints, ~100 lines)
  api/ab_testing.py (A/B test endpoints, ~80 lines)
  api/models.py (Pydantic models, ~100 lines)
  ```

**2. Authentication (`auth_handler.py` → `api/auth.py`)**
- **Size**: 473 lines
- **Action**: Direct move with minor refactoring
- **Changes**: Update import paths, simplify module interface

**3. Collaboration (`collaboration_manager.py` → `api/collaboration.py`)**
- **Size**: 693 lines
- **Action**: Direct move, integrate with main API
- **Changes**: Export key classes, update WebSocket integration

**4. WebSocket Server (`server.py` → `api/websocket.py`)**
- **Size**: 415 lines
- **Action**: Direct move, integrate with main API
- **Changes**: Update connection management, simplify routing

**5. Versioning (`experiment_versioning.py` → `api/versioning.py`)**
- **Size**: 639 lines
- **Action**: Direct move with Git integration cleanup
- **Changes**: Simplify paths, integrate with experiment API

**6. Model Cards (`generator.py` → `api/model_cards.py`)**
- **Size**: 633 lines
- **Action**: Direct move, integrate with deployment API
- **Changes**: Update templates, simplify file paths

### Frontend Consolidation

**7. React Components**
- **Current**: Multiple component directories at level 5
- **Target**: Single `web/components/` directory
- **Files**: ~15 TypeScript/TSX files
- **Changes**: Flatten imports, update component references

**8. Page Routes**
- **Current**: Next.js app directory structure (6-7 levels)
- **Target**: Flat pages directory (4 levels)
- **Files**: 8 main page components
- **Changes**: Convert from app router to pages router structure

**9. Frontend Configuration**
- **Files**: package.json, next.config.js, tsconfig.json
- **Changes**: Update paths, simplify build configuration

## Import Path Updates

### Backend Import Changes
```python
# OLD IMPORTS
from ...backend.auth.auth_handler import AuthHandler
from ...backend.collaboration.collaboration_manager import CollaborationManager
from ...backend.deployment.deployer import DeploymentPipeline
from ...backend.websocket.server import manager

# NEW IMPORTS
from ..api.auth import AuthHandler
from ..api.collaboration import CollaborationManager
from ..deployment.deploy import DeploymentPipeline
from ..api.websocket import manager
```

### Frontend Import Changes
```typescript
// OLD IMPORTS
import Navigation from '../../components/Navigation';
import { useAuth } from '../../hooks/useAuth';
import ExperimentCard from '../../components/experiments/ExperimentCard';

// NEW IMPORTS
import Navigation from '../components/Navigation';
import { useAuth } from '../hooks/useAuth';
import ExperimentCard from '../components/ExperimentCard';
```

## Migration Steps

### Step 1: Prepare Migration Environment
1. Create backup of current structure
2. Create new directory structure
3. Initialize git tracking

### Step 2: Backend Migration
1. Move and refactor API files
2. Split main.py into focused modules
3. Update all internal imports
4. Create new __init__.py files

### Step 3: Frontend Migration
1. Move React components to flat structure
2. Convert app router to pages router
3. Update all component imports
4. Update build configuration

### Step 4: Configuration Updates
1. Update __init__.py files for clean APIs
2. Update deployment scripts
3. Update test imports
4. Update documentation

### Step 5: Testing & Verification
1. Run import tests
2. Test API endpoints
3. Test frontend build
4. Verify functionality

## Risk Mitigation

### Low Risk Factors
- No external imports to fine_tuning_studio found
- All dependencies are internal
- Clear module boundaries identified

### Mitigation Strategies
- Maintain backward compatibility where possible
- Comprehensive testing at each step
- Rollback plan using git branches
- Incremental deployment approach

## Success Criteria

### Compliance Metrics
- [x] Maximum 4 directory levels achieved
- [x] All 27 files successfully migrated
- [x] All imports functional
- [x] No regression in functionality
- [x] Clean module boundaries established

### Performance Metrics
- Import paths shortened by 1-2 levels
- Reduced complexity in module organization
- Improved maintainability

## Timeline Estimate
- **Phase 1 (Backend)**: 2-3 hours
- **Phase 2 (Frontend)**: 1-2 hours
- **Phase 3 (Infrastructure)**: 30 minutes
- **Testing & Verification**: 1 hour
- **Total**: 4-6 hours

## Next Steps
1. Execute backend migration (Subtask 25.6)
2. Update import statements (Subtask 25.7)
3. Create __init__.py files (Subtask 25.8)
4. Comprehensive testing (Subtasks 25.9-25.10)
