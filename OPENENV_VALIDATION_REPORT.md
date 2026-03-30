# URBANEX OpenEnv Validation Report

**Generated:** 2025-01-28  
**Project:** URBANEX (Urban + Exploration) RL Environment  
**OpenEnv Version:** 0.2.3  
**Status:** ⚠️ **PARTIAL COMPLIANCE** (6/10 requirements met)

---

## Summary

Your URBANEX project has a solid foundation but **requires 4 critical structural changes** to achieve full OpenEnv compliance:

1. ✅ **openenv.yaml** exists but **lacks required manifest fields**
2. ✅ **Dockerfile** exists at root
3. ❌ **client.py** is MISSING (blocks deployment)
4. ✅ **models.py** - Observations/Actions exist but incorrectly located
5. ✅ **pyproject.toml** exists but missing [project.scripts]
6. ❌ **server/app.py** doesn't exist in expected location (api/server.py exists instead  
7. ❌ **outputs/** directory missing (recommended)
8. ✅ **__init__.py** exists
9. ✅ **README.md** exists
10. ⚠️  **Manifest fields** - Missing client/action/observation sections

---

## Validation Details

### 1. CRITICAL ISSUES (Must Fix)

#### A. Missing `client.py` 🔴
**Status:** ❌ BLOCKING  
**Impact:** Cannot create EnvClient instances; deployment will fail  
**Location:** Should be at `/client.py` (root)  
**Current:** Only `api/server.py` exists

```
Example client.py structure needed:
└── client.py
    └── class UrbanexEnv(EnvClient): ...
```

**Fix Required:** Create wrapper client that inherits from `openenv.core.EnvClient`

---

#### B. Missing `models.py` at Root 🔴
**Status:** ❌ CRITICAL  
**Impact:** Module imports `from urbanex.models import Action, Observation` will fail  
**Current Location:** `models/action.py`, `models/observation.py`  
**Expected Location:** `/models.py` or structured properly with `__init__.py`

**Current Structure:**
```
models/
  ├── __init__.py (empty?)
  ├── action.py      ✅ Has Action dataclass
  ├── observation.py ✅ Has Observation dataclass  
  ├── reward.py
  └── ...
```

**What Needs to Happen:** Either
- Option A: Create `/models.py` that re-exports Action, Observation
- Option B: Ensure `models/__init__.py` properly exports them

---

#### C. Invalid `openenv.yaml` Manifest 🔴
**Status:** ❌ CRITICAL  
**Impact:** OpenEnv CLI `validate` and `push` commands will fail  

**Current Format:** Contains detailed schema but MISSING required manifest sections

**Current Missing Sections:**
```yaml
# MISSING - Must be added:
client:
  class_name: UrbanexEnv
  module: urbanex.client

action:
  class_name: Action
  module: urbanex.models

observation:
  class_name: Observation
  module: urbanex.models
```

**What's There:** ✅ 
- name, version, description
- domain, tasks, action_space, observation_space
- (detailed JSON schema - good for documentation, not for manifest validation)

**What's Missing:** ❌
- `client` section with class_name and module
- `action` section with class_name and module  
- `observation` section with class_name and module
- `spec_version` (recommended: 1)
- `default_image` (recommended)

---

#### D. Server Structure Mismatch 🟡
**Status:** ⚠️ NEEDS REORGANIZATION  
**Current:** `api/server.py` (FastAPI server exists here)  
**Expected:** `server/app.py` with `main()` function

**Validation Requirements:**
- ✅ server/Dockerfile exists? → Check if you have root Dockerfile instead
- ❌ server/__init__.py exists? → MISSING
- ❌ server/app.py exists? → Currently at api/server.py (wrong location)

**Options to Fix:**
1. Move FastAPI server to `server/app.py` and ensure `main()` function exists
2. OR: Keep `api/server.py` but create wrapper at `server/app.py` that imports from it

---

### 2. WARNINGS (Should Fix)

#### A. Missing `outputs/` Directory 🟡
**Status:** ⚠️ RECOMMENDED  
**Impact:** OpenEnv expects outputs, logs, evals subdirectories  
**Fix:** Create these directories (gitignored):
```
outputs/
  ├── logs/
  └── evals/
```

---

#### B. `pyproject.toml` Missing [project.scripts] 🟡
**Status:** ⚠️ NEEDED FOR LOCAL DEVELOPMENT  
**Current:** No scripts section  
**Expected for Development:**
```toml
[project.scripts]
urbanex = "api.server:main"
```

**Or if reorganizing to server/app.py:**
```toml
[project.scripts]
urbanex = "server.app:main"
```

---

### 3. PASSING VALIDATIONS ✅

| Check | Status | Details |
|-------|--------|---------|
| openenv.yaml exists | ✅ | Present at root |
| __init__.py exists | ✅ | Present at root |
| README.md exists | ✅ | Present at root |
| Dockerfile exists | ✅ | Present at root |
| pyproject.toml exists | ✅ | Well-structured |
| Python version >= 3.10 | ✅ | Requires 3.11 |
| FastAPI in deps | ✅ | fastapi==0.111.0 |
| Uvicorn in deps | ✅ | uvicorn==0.29.0 |
| Pydantic in deps | ✅ | pydantic==2.7.1 |

---

## OpenEnv Manifest Specification

**The `openenv.yaml` file must follow this schema:**

```yaml
# REQUIRED
name: urbanex                    # Environment unique identifier
version: "1.0.0"                 # Semantic version
description: >                   # Brief description
  URBANEX - Urban navigation RL environment

# REQUIRED - Client configuration
client:
  class_name: UrbanexEnv         # Class name of EnvClient subclass
  module: urbanex.client         # Python module path

# REQUIRED - Action type
action:
  class_name: Action             # Class name of Pydantic Action dataclass
  module: urbanex.models         # Python module path

# REQUIRED - Observation type  
observation:
  class_name: Observation        # Class name of Pydantic Observation dataclass
  module: urbanex.models         # Python module path

# RECOMMENDED
spec_version: 1                  # OpenEnv spec version
default_image: urbanex:latest    # Default Docker image name

# OPTIONAL - For documentation (not used by OpenEnv CLI)
# domain: urban-mobility         # Can stay, but not validated
# tasks: [...]                   # Can stay, but not validated
# action_space: {...}            # Can stay, but not validated  
# observation_space: {...}       # Can stay, but not validated
```

---

## Validation Checklist

### Before `openenv validate` will pass:

- [ ] **client.py** created with UrbanexEnv(EnvClient) class
- [ ] **models.py** has Action, Observation exported (or models/ properly configured)
- [ ] **openenv.yaml** updated with client/action/observation manifest sections
- [ ] **server/app.py** exists with `main()` function (or api config updated)
- [ ] **server/__init__.py** exists (if using server/ structure)
- [ ] **pyproject.toml** has [project.scripts] entry
- [ ] **outputs/** directory created

### Before `openenv push` will pass:

- [ ] All items above
- [ ] FastAPI server returns `/metadata` endpoint
- [ ] FastAPI server returns `/schema` endpoint
- [ ] WebSocket endpoint `/ws` works
- [ ] `/health` endpoint returns 200

---

## Directory Structure After Fixes

```
URBANEX/
├── openenv.yaml                    # ✏️ UPDATE
├── pyproject.toml                  # ✏️ UPDATE  
├── Dockerfile                      # ✅ Exists
├── README.md                       # ✅ Exists
├── __init__.py                     # ✅ Exists
├── client.py                       # 🆕 CREATE
├── models.py                       # 🆕 CREATE (or fix __init__.py)
├── outputs/                        # 🆕 CREATE
│   ├── logs/
│   └── evals/
├── api/
│   ├── __init__.py
│   └── server.py                   # ✅ Exists - keep for now
├── environment/
│   ├── __init__.py
│   ├── velora_env.py              # Main environment class
│   ├── city.py
│   ├── incidents.py
│   ├── rewards.py
│   ├── routes.py
│   └── __pycache__/
├── baseline/
│   ├── __init__.py
│   ├── baseline_agent.py
│   ├── visualizer.py
│   └── __pycache__/
├── graders/
│   ├── __init__.py
│   ├── grader_easy.py
│   ├── grader_medium.py
│   ├── grader_hard.py
│   └── __pycache__/
├── models/
│   ├── __init__.py                # ✏️ FIX - Export Action, Observation
│   ├── action.py                  # ✅ Has Action
│   ├── observation.py             # ✅ Has Observation
│   ├── reward.py
│   └── __pycache__/
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py
│   ├── task_medium.py
│   ├── task_hard.py
│   └── __pycache__/
├── tests/
│   ├── test_env.py
│   ├── test_graders.py
│   └── __pycache__/
└── conftest.py

OPTION A: Keep api/server.py (create wrapper)
├── server/                         # 🆕 CREATE
│   ├── __init__.py                # 🆕 CREATE
│   ├── app.py                     # 🆕 CREATE - wrapper importing from api
│   └── requirements.txt           # 🆕 CREATE (or auto-generated)

OPTION B: Reorganize (move to server/)
├── server/
│   ├── __init__.py
│   ├── app.py                     # Move/refactor from api/server.py
│   ├── velora_env_server.py       # Main server logic
│   ├── Dockerfile
│   └── requirements.txt
```

---

## Recommended Fixes (In Order)

### Phase 1: Model Exports (5 min)
Update `models/__init__.py`:
```python
from models.action import Action
from models.observation import Observation, RouteOption, Incident

__all__ = ["Action", "Observation", "RouteOption", "Incident"]
```

### Phase 2: Update openenv.yaml Manifest (2 min)
Add required sections after `description`:
```yaml
client:
  class_name: UrbanexEnv
  module: urbanex.client

action:
  class_name: Action
  module: urbanex.models

observation:
  class_name: Observation
  module: urbanex.models

spec_version: 1
default_image: urbanex:latest
```

### Phase 3: Create client.py (10 min)
Create minimal OpenEnv client wrapper around existing server.

### Phase 4: Reorganize Server (15 min)
Move or wrap FastAPI app to `server/app.py` with `main()` function.

### Phase 5: Update pyproject.toml (2 min)
Add [project.scripts] section.

### Phase 6: Create outputs/ (1 min)
```bash
mkdir -p outputs/{logs,evals}
echo "*" > outputs/.gitignore
```

---

## CLI Validation Commands

Once you make the fixes above, test with:

```bash
# Validate local environment structure
openenv validate

# Validate and test runtime (needs server running)
openenv validate --with-runtime

# Push to Hugging Face (once validation passes)
openenv push --repo-id your-username/urbanex-env

# Push to Docker Hub
openenv push --registry docker.io/your-username
```

---

## References

- **OpenEnv GitHub:** https://github.com/meta-pytorch/OpenEnv
- **Validation Source:** https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/cli/_cli_utils.py
- **CLI Documentation:** https://meta-pytorch.org/OpenEnv/cli.html

---

## Next Steps

Ready to implement these fixes? I can help you:

1. ✏️ Create the `client.py` wrapper
2. ✏️ Update `openenv.yaml` manifest
3. ✏️ Reorganize server structure
4. ✏️ Fix dependency configuration
5. ✏️ Run validation tests

What would you like to tackle first?
