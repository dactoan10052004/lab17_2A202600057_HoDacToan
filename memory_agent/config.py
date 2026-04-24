from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USER_ID = os.getenv("USER_ID", "demo_user")
ENABLE_LLM_MEMORY_EXTRACTION = os.getenv("ENABLE_LLM_MEMORY_EXTRACTION", "0") == "1"

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

EPISODE_LOG_PATH = DATA_DIR / "episodes.jsonl"
PROFILE_STORE_PATH = DATA_DIR / "profiles.json"
SEMANTIC_DOC_PATH = DATA_DIR / "semantic_docs.json"
CHROMA_DIR = DATA_DIR / "chroma"

# Demo context budget. Production nên tính theo model context thật.
MAX_CONTEXT_TOKENS = 6000
MEMORY_BUDGET_TOKENS = 1200

# Redis TTLs, theo tinh thần slide lab:
PREF_TTL_SECONDS = 90 * 24 * 3600
FACT_TTL_SECONDS = 30 * 24 * 3600
SESSION_TTL_SECONDS = 7 * 24 * 3600
