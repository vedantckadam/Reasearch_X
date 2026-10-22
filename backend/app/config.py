import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Define absolute paths independent of current working directory
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT_DIR = BACKEND_DIR.parent

BACKEND_ENV_PATH = BACKEND_DIR / ".env"
PROJECT_ROOT_ENV_PATH = PROJECT_ROOT_DIR / ".env"

KNOWN_PLACEHOLDERS = {
    "your_actual_key_goes_here",
    "your_api_key_here",
    "your_openrouter_api_key_here",
    "your_nvidia_api_key_here",
    "your_key_here",
    "replace_with_your_key",
    "",
}

def resolve_env_path() -> Optional[Path]:
    """Finds the active .env file by checking backend/.env then root .env."""
    if BACKEND_ENV_PATH.is_file():
        return BACKEND_ENV_PATH
    if PROJECT_ROOT_ENV_PATH.is_file():
        return PROJECT_ROOT_ENV_PATH
    return None

def load_project_env(override: bool = False) -> Optional[Path]:
    """
    Loads environment variables from backend/.env (or root .env) using absolute paths.
    If override=True, re-reads values from the file to pick up modifications made
    while long-running servers like Uvicorn are active.
    Preserves mocked test environments when executing under pytest.
    """
    if override and os.getenv("PYTEST_CURRENT_TEST"):
        return resolve_env_path()
    env_path = resolve_env_path()
    if env_path:
        load_dotenv(dotenv_path=env_path, override=override)
        return env_path
    load_dotenv(override=override)
    return None

# Initial load on import
_ACTIVE_ENV_PATH = load_project_env(override=False)

def is_key_configured(raw_key: Optional[str]) -> bool:
    """Safely checks if an API key is present and not a known placeholder."""
    if not raw_key:
        return False
    clean = raw_key.strip("'\" \t\r\n")
    return bool(clean and clean.lower() not in KNOWN_PLACEHOLDERS)

def get_llm_diagnostic() -> Dict[str, Any]:
    """
    Returns diagnostic status of LLM configuration without ever revealing
    the API key or any slice of secret credentials.
    """
    provider = (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
    raw_key = os.getenv("OPENROUTER_API_KEY" if provider == "openrouter" else "NVIDIA_NEMOTRON_API_KEY") or ""
    
    # If not configured, attempt a reload from disk to pick up any recent file edits
    env_path = resolve_env_path()
    if not is_key_configured(raw_key):
        load_project_env(override=True)
        raw_key = os.getenv("OPENROUTER_API_KEY" if provider == "openrouter" else "NVIDIA_NEMOTRON_API_KEY") or ""
    
    if provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL") or "nvidia/nemotron-3-ultra-550b-a55b:free"
        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    else:
        model = os.getenv("NVIDIA_NEMOTRON_MODEL") or "nvidia/nemotron-4-340b-instruct"
        base_url = (os.getenv("NVIDIA_NEMOTRON_BASE_URL") or "https://integrate.api.nvidia.com/v1").rstrip("/")

    clean_key = raw_key.strip("'\" \t\r\n")
    key_present = bool(clean_key)
    is_placeholder = bool(clean_key.lower() in KNOWN_PLACEHOLDERS) if key_present else True
    configured = key_present and not is_placeholder

    return {
        "status": "configured" if configured else "unconfigured",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "endpoint": f"{base_url}/chat/completions",
        "configured": configured,
        "key_present": key_present,
        "is_placeholder": is_placeholder,
        "env_file_detected": env_path.name if env_path else None,
        "env_file_location": str(env_path) if env_path else None,
    }
