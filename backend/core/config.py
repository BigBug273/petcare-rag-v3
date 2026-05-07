"""Backend configuration and project paths."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
INDEX_HTML = FRONTEND_STATIC_DIR / "index.html"

load_dotenv(PROJECT_ROOT / ".env")
