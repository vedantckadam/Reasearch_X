#!/usr/bin/env python3
r"""
Standalone Live Connection Test for OpenRouter with NVIDIA Nemotron.

This script executes a real (non-mocked) HTTP request to the OpenRouter API
using the configured backend settings.

Usage:
    cd D:\My_projects\Reasearch_GapX\backend
    .\venv\Scripts\activate
    python scripts/test_openrouter_connection.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv

# Explicitly load backend .env
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from app.services.nemotron_client import NemotronClient

KNOWN_PLACEHOLDERS = {
    "your_actual_key_goes_here",
    "your_api_key_here",
    "your_openrouter_api_key_here",
    "your_nvidia_api_key_here",
    "",
}

def mask_presence(val: str) -> str:
    """Safely reports key presence without revealing characters."""
    if not val or val.strip() in KNOWN_PLACEHOLDERS:
        return "[NOT SET / PLACEHOLDER]"
    return f"[CONFIGURED - {len(val.strip())} chars]"

def main() -> int:
    print("=" * 65)
    print(" ResearchGap-X: OpenRouter API Live Connection Verification")
    print("=" * 65)

    provider = (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    model = os.getenv("OPENROUTER_MODEL") or "nvidia/nemotron-3.5-lightning:free"
    api_key = os.getenv("OPENROUTER_API_KEY") or ""

    print(f"Provider:        {provider}")
    print(f"Base URL:        {base_url}")
    print(f"Endpoint:        {base_url}/chat/completions")
    print(f"Configured Model:{model}")
    print(f"API Key Status:  {mask_presence(api_key)}")
    print("-" * 65)

    # 1 & 2: Fail safely if key is missing or placeholder
    if not api_key or api_key.strip() in KNOWN_PLACEHOLDERS:
        print("\n[FAILED] Configuration Error:")
        print("  OPENROUTER_API_KEY is not configured or still contains a placeholder.")
        print("\nTo resolve this:")
        print("  1. Open 'backend/.env' in your editor.")
        print("  2. Replace 'your_actual_key_goes_here' with your real OpenRouter API key:")
        print("     OPENROUTER_API_KEY=sk-or-v1-...")
        print("  3. Run this script again:")
        print("     python scripts/test_openrouter_connection.py\n")
        return 1

    print("Initializing NemotronClient(provider='openrouter')...")
    client = NemotronClient(provider="openrouter")

    test_messages = [
        {
            "role": "user",
            "content": "Reply with only the word: CONNECTED",
        }
    ]

    print(f"Sending real request to {model} at {base_url}/chat/completions...")
    start_time = time.time()

    try:
        response = client.post_chat_completion(
            messages=test_messages,
            temperature=0.1,
            max_tokens=30,
        )
        elapsed_s = time.time() - start_time

        content = (response.get("content") or "").strip()
        returned_model = response.get("model") or model
        req_id = response.get("id") or "N/A"

        print("-" * 65)
        print("HTTP Status:     200 OK")
        print(f"Latency:         {elapsed_s:.2f} seconds")
        print(f"Model Reported:  {returned_model}")
        print(f"Response ID:     {req_id}")
        print(f"Excerpt Answer:  {content[:120]}")
        print("-" * 65)

        # 5: Confirm non-empty model response
        if not content:
            print("[FAILED] Model returned an empty response content.")
            return 1

        print("\n[SUCCESS] OpenRouter connection successfully verified!")
        print(f"The model '{model}' responded with valid non-empty content.")
        print("=" * 65)
        return 0

    except PermissionError as e:
        print("-" * 65)
        print("[FAILED] Authentication Error (HTTP 401):")
        print(f"  {str(e)}")
        print("  Check that your OPENROUTER_API_KEY is valid and has not expired or been revoked.")
        print("=" * 65)
        return 1

    except LookupError as e:
        print("-" * 65)
        print("[FAILED] Model Availability Error (HTTP 404):")
        print(f"  {str(e)}")
        print(f"  The model '{model}' was not found on OpenRouter.")
        print("  Verify model identifier at https://openrouter.ai/models.")
        print("=" * 65)
        return 1

    except TimeoutError as e:
        print("-" * 65)
        print("[FAILED] Request Timeout:")
        print(f"  {str(e)}")
        print("  The OpenRouter server or upstream provider did not respond in time.")
        print("  Consider increasing LLM_TIMEOUT in backend/.env.")
        print("=" * 65)
        return 1

    except ConnectionError as e:
        print("-" * 65)
        print("[FAILED] Network / Connection Error:")
        print(f"  {str(e)}")
        print("  Could not reach OpenRouter. Check your internet connection or proxy settings.")
        print("=" * 65)
        return 1

    except RuntimeError as e:
        print("-" * 65)
        err_str = str(e)
        if "429" in err_str:
            print("[FAILED] Rate Limit Exceeded (HTTP 429):")
            print(f"  {err_str}")
            print("  The free tier rate limit was hit. Wait a few moments before retrying.")
        elif "402" in err_str:
            print("[FAILED] Insufficient Credits / Payment Required (HTTP 402):")
            print(f"  {err_str}")
            print("  Check your OpenRouter account balance at https://openrouter.ai/credits.")
        else:
            print("[FAILED] API Request Failed:")
            print(f"  {err_str}")
        print("=" * 65)
        return 1

    except Exception as e:
        print("-" * 65)
        print(f"[FAILED] Unexpected Error: {str(e)}")
        print("=" * 65)
        return 1

if __name__ == "__main__":
    sys.exit(main())
