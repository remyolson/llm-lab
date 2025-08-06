# Fine-Tuning Studio Directory Restructuring Plan

## Current Issues
- Maximum nesting depth: 6 levels (exceeds 4-level requirement)
- Duplicate functionality between `fine_tuning` and `fine_tuning_studio`
- Mixed backend/frontend concerns in single module tree
- Poor separation of concerns

## Current Structure
```
src/use_cases/fine_tuning_studio/                              # Level 3
├── backend/                                                   # Level 4
│   ├── api/main.py                                           # Level 5
│   ├── auth/auth_handler.py                                  # Level 5
│   ├── collaboration/collaboration_manager.py               # Level 5
│   ├── deployment/deployer.py                               # Level 5
│   ├── model_card/generator.py                              # Level 5
│   ├── testing/test_suite.py                                # Level 5
│   ├── versioning/experiment_versioning.py                  # Level 5
│   └── websocket/server.py                                  # Level 5
├── deployment/                                               # Level 4
│   ├── configs/                                             # Level 5
│   └── scripts/                                             # Level 5
└── frontend/                                                 # Level 4
    ├── app/                                                  # Level 5
    │   ├── dashboard/page.tsx                               # Level 6 ❌
    │   ├── experiments/new/page.tsx                         # Level 7 ❌❌
    │   └── [other routes]                                   # Level 6 ❌
    ├── components/                                          # Level 5
    ├── hooks/                                               # Level 5
    └── [other dirs]                                         # Level 5
```

## Proposed New Structure (Max 4 levels)
```
src/use_cases/fine_tuning/                                    # Level 3
├── [existing modules unchanged]                              # Level 4
├── api/                                                      # Level 4 ✅
│   ├── main.py                                              # API server
│   ├── auth.py                                              # Authentication
│   ├── experiments.py                                       # Experiment endpoints
│   ├── datasets.py                                          # Dataset endpoints
│   ├── deployments.py                                       # Deployment endpoints
│   ├── websocket.py                                         # WebSocket server
│   └── models.py                                            # Pydantic models
├── web/                                                      # Level 4 ✅
│   ├── components.py                                        # React component defs
│   ├── pages.py                                             # Page definitions
│   ├── hooks.py                                             # React hooks
│   ├── types.py                                             # TypeScript types
│   └── static/                                              # Static assets
└── deployment/                                               # Level 4 ✅
    ├── deploy.py                                            # Deployment logic
    ├── configs.py                                           # Configuration
    └── scripts.py                                           # Deployment scripts
```

## Migration Strategy

### Phase 1: Flatten Backend Structure
- Move `backend/api/main.py` → `api/main.py`
- Consolidate `backend/auth/`, `backend/collaboration/`, etc. into focused modules
- Move `backend/websocket/server.py` → `api/websocket.py`

### Phase 2: Consolidate Frontend
- Move frontend components to `web/` directory
- Flatten React app structure
- Consolidate TypeScript definitions

### Phase 3: Simplify Deployment
- Move deployment scripts to `deployment/` directory
- Consolidate configuration management

### Phase 4: Update Imports
- Update all import statements
- Create proper `__init__.py` files
- Ensure backward compatibility where possible

## Benefits
1. **Compliance**: Meets 4-level maximum depth requirement
2. **Consolidation**: Eliminates duplication with existing `fine_tuning` module
3. **Clarity**: Clear separation between API, web, and deployment concerns
4. **Maintainability**: Easier to navigate and understand
5. **Standards**: Follows Python packaging conventions

## File Movement Summary
- 27 files to be moved/consolidated
- 8 directories to be flattened
- 3 new directories created (`api/`, `web/`, `deployment/`)
- 9 old directories removed

## Impact Assessment
- **Low Risk**: No external imports found to `fine_tuning_studio`
- **Medium Effort**: Requires updating internal imports and paths
- **High Value**: Significantly improves codebase organization
