
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
TEST_ENV = BASE_DIR / ".env.test"

if not TEST_ENV.is_file():
    raise SystemExit("Missing backend/.env.test")

# Load test settings before importing the application.
# Override any production variables already in this terminal.
load_dotenv(TEST_ENV, override=True)

required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD", "GEMINI_API_KEY"]
missing = [key for key in required if not os.getenv(key)]

if missing:
    raise SystemExit(
        "Missing required settings in backend/.env.test: "
        + ", ".join(missing)
    )

# Ensure imports work when launched from the project root.
sys.path.insert(0, str(BASE_DIR))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )
