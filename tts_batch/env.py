"""Loads a .env file's KEY=VALUE lines into os.environ, if one exists.

Generic on purpose: any backend that reads a secret via os.environ (today
just CARTESIA_API_KEY, in backends/cartesia.py) picks it up automatically,
with no per-key wiring needed here or for future keys a new backend might
add. Real environment variables always win over .env - .env only fills in
what isn't already set, so `CARTESIA_API_KEY=... python generate_and_concat.py
...` still overrides a .env file.
"""

import os
from pathlib import Path

DOTENV_PATH = Path(".env")


def load_dotenv(path: Path = DOTENV_PATH) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        os.environ.setdefault(key, value)
