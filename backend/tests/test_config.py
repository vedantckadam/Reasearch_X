import os
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import (
    resolve_env_path,
    is_key_configured,
    get_llm_diagnostic,
    load_project_env,
    BACKEND_ENV_PATH,
)

client = TestClient(app)

def test_resolve_env_path_finds_backend_env():
    path = resolve_env_path()
    assert path is not None
    assert path.name == ".env"
    assert path.is_file()

def test_is_key_configured():
    # Empty and None
    assert is_key_configured(None) is False
    assert is_key_configured("") is False
    assert is_key_configured("   ") is False

    # Known placeholders
    assert is_key_configured("your_actual_key_goes_here") is False
    assert is_key_configured("your_openrouter_api_key_here") is False
    assert is_key_configured("your_api_key_here") is False
    assert is_key_configured('"your_actual_key_goes_here"') is False
    assert is_key_configured("'your_openrouter_api_key_here'") is False

    # Valid mocked key
    assert is_key_configured("sk-or-v1-testkey123456789") is True
    assert is_key_configured('"sk-or-v1-testkey123456789"') is True

@patch.dict(os.environ, {"OPENROUTER_API_KEY": "your_openrouter_api_key_here", "LLM_PROVIDER": "openrouter"})
def test_diagnostic_with_placeholder():
    diag = get_llm_diagnostic()
    assert diag["configured"] is False
    assert diag["is_placeholder"] is True
    assert diag["provider"] == "openrouter"
    assert diag["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    # Crucial: verify key is NEVER in output
    assert "OPENROUTER_API_KEY" not in diag
    assert "your_openrouter_api_key_here" not in str(diag)

@patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-mock-secret-key-xyz", "LLM_PROVIDER": "openrouter"})
def test_diagnostic_with_mock_real_key():
    diag = get_llm_diagnostic()
    assert diag["configured"] is True
    assert diag["key_present"] is True
    assert diag["is_placeholder"] is False
    assert diag["provider"] == "openrouter"
    # Crucial: verify real secret is NEVER leaked
    assert "sk-or-v1-mock-secret-key-xyz" not in str(diag)

def test_api_llm_status_endpoint():
    resp = client.get("/api/llm/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data
    assert "model" in data
    assert "base_url" in data
    assert "endpoint" in data
    # Ensure no secret value leak in response
    assert "api_key" not in data
