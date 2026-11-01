"""
Centralized Environment Configuration Loader.

Guarantees environment files (.env and api.env) are loaded from the exact project root
directory, regardless of the process working directory or invocation path.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Calculate the project root directory based on this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_project_env() -> None:
    """
    Load environment variables from project root .env and api.env files.
    Precedence: .env overrides or supplements api.env if both exist.
    """
    api_env_path = PROJECT_ROOT / "api.env"
    env_path = PROJECT_ROOT / ".env"

    if api_env_path.is_file():
        load_dotenv(dotenv_path=str(api_env_path), override=False)

    if env_path.is_file():
        load_dotenv(dotenv_path=str(env_path), override=True)


def get_env_var(name: str, default: str = None) -> str:
    """
    Safely retrieve an environment variable from os.environ after ensuring
    project environment files have been loaded.
    """
    load_project_env()
    val = os.getenv(name, default)
    if val is not None:
        val = val.strip()
    return val


# Automatically load on import
load_project_env()

