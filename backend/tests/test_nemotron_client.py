import os
import json
import pytest
from unittest.mock import patch, MagicMock
import requests
from app.services.nemotron_client import NemotronClient

VALID_EXPLANATION_DATA = {
    "candidate_gap_statement": "Integration of differential privacy in edge federated networks.",
    "evidence_summary": "Found 2 limitation snippets pointing to communication bottlenecks.",
    "literature_context": "Recent surveys explore local DP but lack communication bounds.",
    "limitation_evidence": "Explicit author limitations regarding transmission cost.",
    "temporal_context": "Topic activity has increased 40% since 2022.",
    "counterevidence": "Centralized DP solutions exist but do not meet edge constraints.",
    "evidence_limitations": "Evaluated on 100 corpus papers.",
    "suggested_direction": "Develop adaptive noise injection schemes for heterogeneous clients.",
}

@patch.dict(os.environ, {"NVIDIA_NEMOTRON_API_KEY": "fake_key", "LLM_PROVIDER": "nvidia"})
@patch("requests.post")
def test_nemotron_nvidia_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "gen-123",
        "model": "nvidia/nemotron-4-340b-instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(VALID_EXPLANATION_DATA)
                },
                "finish_reason": "stop"
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="nvidia")
    packet = {
        "candidate_id": "c1",
        "source_metadata": [
            {"paper_id": "p1", "title": "T1", "year": 2020, "chunk_id": "c1", "source_field": "abstract"}
        ]
    }

    res = client.generate_grounded_explanation("Prompt", packet)
    assert res.candidate_gap_statement == VALID_EXPLANATION_DATA["candidate_gap_statement"]
    assert res.suggested_direction == VALID_EXPLANATION_DATA["suggested_direction"]
    assert res.grounding_status == "Success"
    assert len(res.source_references) == 1
    assert res.source_references[0].paper_id == "p1"
    assert client.provider == "nvidia"

@patch.dict(os.environ, {"NVIDIA_NEMOTRON_API_KEY": "", "LLM_PROVIDER": "nvidia"})
def test_nemotron_nvidia_missing_key():
    client = NemotronClient(provider="nvidia")
    with pytest.raises(ValueError, match="not configured"):
        client.generate_grounded_explanation("Prompt", {})

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_or_key",
    "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
})
@patch("requests.post")
def test_openrouter_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "gen-or-456",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f'```json\n{json.dumps(VALID_EXPLANATION_DATA)}\n```'
                },
                "finish_reason": "stop"
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    assert client.provider == "openrouter"
    assert client.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert client.base_url == "https://openrouter.ai/api/v1"

    packet = {
        "candidate_id": "cand_or_1",
        "source_metadata": [
            {"paper_id": "p2", "title": "Paper 2", "year": 2024, "chunk_id": "ch2", "source_field": "abstract"}
        ]
    }

    res = client.generate_grounded_explanation("Prompt", packet)
    assert res.candidate_id == "cand_or_1"
    assert res.candidate_gap_statement == VALID_EXPLANATION_DATA["candidate_gap_statement"]
    assert res.suggested_direction == VALID_EXPLANATION_DATA["suggested_direction"]
    assert res.grounding_status == "Success"
    assert len(res.source_references) == 1
    assert res.source_references[0].paper_id == "p2"

    # Verify headers sent to OpenRouter
    call_kwargs = mock_post.call_args.kwargs
    headers = call_kwargs["headers"]
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer fake_or_key"
    assert headers["HTTP-Referer"] == "https://github.com/researchgapx"
    assert headers["X-Title"] == "ResearchGap-X"
    assert call_kwargs["json"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
}, clear=True)
def test_openrouter_default_model():
    client = NemotronClient(provider="openrouter")
    assert client.model == "nvidia/nemotron-3-ultra-550b-a55b:free"

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "",
})
def test_openrouter_missing_key():
    client = NemotronClient(provider="openrouter")
    assert not client.is_configured()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        client.post_chat_completion([{"role": "user", "content": "hi"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "your_actual_key_goes_here",
})
def test_openrouter_placeholder_key_actual():
    client = NemotronClient(provider="openrouter")
    assert not client.is_configured()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        client.post_chat_completion([{"role": "user", "content": "hi"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "your_openrouter_api_key_here",
})
def test_openrouter_placeholder_key_named():
    client = NemotronClient(provider="openrouter")
    assert not client.is_configured()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        client.post_chat_completion([{"role": "user", "content": "hi"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_secret_key_123",
})
@patch("requests.post")
def test_openrouter_401_auth_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": {"message": "Invalid API key provided", "code": 401}}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(PermissionError) as exc_info:
        client.post_chat_completion([{"role": "user", "content": "hello"}])
    # Ensure raw API key is never leaked in the exception
    assert "fake_secret_key_123" not in str(exc_info.value)
    assert "HTTP 401" in str(exc_info.value)

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_secret_key_123",
})
@patch("requests.post")
def test_openrouter_403_forbidden_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {"error": {"message": "Access forbidden", "code": 403}}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(PermissionError) as exc_info:
        client.post_chat_completion([{"role": "user", "content": "hello"}])
    assert "fake_secret_key_123" not in str(exc_info.value)
    assert "HTTP 403" in str(exc_info.value)

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_402_insufficient_credits(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.json.return_value = {"error": {"message": "Insufficient credits", "code": 402}}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(RuntimeError, match="HTTP 402"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_404_model_not_found(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"error": {"message": "Model not found", "code": 404}}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(LookupError, match="HTTP 404"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_429_rate_limit(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded", "code": 429}}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(RuntimeError, match="HTTP 429"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_5xx_server_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"error": {"message": "Backend service unavailable", "code": 503}}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    client = NemotronClient(provider="openrouter", timeout=5)
    with pytest.raises(TimeoutError, match="timed out"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = NemotronClient(provider="openrouter")
    with pytest.raises(ConnectionError, match="Failed to connect"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_invalid_json_response(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON in body")
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(ValueError, match="Failed to decode JSON"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_empty_choices(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": []}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(ValueError, match="empty 'choices' list"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_missing_choices(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "no-choices"}
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(ValueError, match="missing 'choices'"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_missing_content(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": ""}
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(ValueError, match="empty message content"):
        client.post_chat_completion([{"role": "user", "content": "hello"}])

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
})
@patch("requests.post")
def test_openrouter_llm_invalid_json_generation(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "I cannot generate valid JSON for this query."}
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    with pytest.raises(ValueError, match="Could not parse valid JSON"):
        client.generate_grounded_explanation("Prompt", {"candidate_id": "c1"})

@patch.dict(os.environ, {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "fake_key",
    "OPENROUTER_REASONING_EFFORT": "medium",
})
@patch("requests.post")
def test_openrouter_reasoning_parameter(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Answer with reasoning",
                    "reasoning_details": "Thinking step by step..."
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = NemotronClient(provider="openrouter")
    res = client.post_chat_completion([{"role": "user", "content": "think"}])

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["reasoning"] == {"effort": "medium"}
    # Verify content contains user-facing text and reasoning_details is preserved separately
    assert res["content"] == "Answer with reasoning"
    assert res["reasoning_details"] == "Thinking step by step..."


# =====================================================================
# Regression Tests: Robust JSON Parsing & Schema Validation (10 Suites)
# =====================================================================

def test_regression_1_valid_plain_json():
    """Scenario 1: Valid plain JSON object with all required fields."""
    client = NemotronClient(api_key="mock_key")
    raw = json.dumps(VALID_EXPLANATION_DATA)
    parsed = client._parse_json_content(raw)
    assert parsed == VALID_EXPLANATION_DATA
    for field in VALID_EXPLANATION_DATA:
        assert field in parsed
        assert isinstance(parsed[field], str)


def test_regression_2_json_in_markdown_fences():
    """Scenario 2: JSON wrapped in markdown code fences (```json and ```)."""
    client = NemotronClient(api_key="mock_key")
    # With language identifier
    fenced_json = f"```json\n{json.dumps(VALID_EXPLANATION_DATA, indent=2)}\n```"
    parsed1 = client._parse_json_content(fenced_json)
    assert parsed1 == VALID_EXPLANATION_DATA

    # Without language identifier
    fenced_plain = f"```\n{json.dumps(VALID_EXPLANATION_DATA)}\n```"
    parsed2 = client._parse_json_content(fenced_plain)
    assert parsed2 == VALID_EXPLANATION_DATA


def test_regression_3_json_with_surrounding_harmless_text():
    """Scenario 3: Harmless preamble and postamble text surrounding valid JSON."""
    client = NemotronClient(api_key="mock_key")
    wrapped = (
        "Here is the grounded research gap analysis synthesized from your corpus:\n\n"
        f"{json.dumps(VALID_EXPLANATION_DATA)}\n\n"
        "Let me know if you need additional evidence extraction!"
    )
    parsed = client._parse_json_content(wrapped)
    assert parsed == VALID_EXPLANATION_DATA


def test_regression_4_truncated_json_unclosed():
    """Scenario 4: Incomplete/truncated JSON missing closing brace."""
    client = NemotronClient(api_key="mock_key")
    truncated = '{"candidate_gap_statement": "The analyzed corpus shows a sparse intersection across 65 documents'
    with pytest.raises(ValueError) as exc_info:
        client._parse_json_content(truncated)
    err = str(exc_info.value)
    assert "Incomplete JSON in LLM response" in err or "truncated" in err
    assert "length=" in err


def test_regression_5_invalid_json_syntax():
    """Scenario 5: Malformed JSON syntax inside object."""
    client = NemotronClient(api_key="mock_key")
    malformed = '{"candidate_gap_statement": "Broken", undefined_field: 123,,}'
    with pytest.raises(ValueError) as exc_info:
        client._parse_json_content(malformed)
    err = str(exc_info.value)
    assert "Could not parse valid JSON from LLM response" in err
    assert "length=" in err
    assert "pos" in err


def test_regression_6_missing_required_fields():
    """Scenario 6: Valid JSON missing required schema fields."""
    client = NemotronClient(api_key="mock_key")
    incomplete = {
        "candidate_gap_statement": "Only candidate statement provided",
        "evidence_summary": "Only evidence summary provided",
    }
    with pytest.raises(ValueError) as exc_info:
        client._parse_json_content(json.dumps(incomplete))
    err = str(exc_info.value)
    assert "missing required fields" in err
    assert "literature_context" in err
    assert "limitation_evidence" in err
    assert "suggested_direction" in err


def test_regression_7_incorrect_field_types():
    """Scenario 7: JSON with non-string field types."""
    client = NemotronClient(api_key="mock_key")
    wrong_types = dict(VALID_EXPLANATION_DATA)
    wrong_types["evidence_summary"] = 12345  # int instead of str
    wrong_types["counterevidence"] = ["list", "of", "items"]  # list instead of str

    with pytest.raises(ValueError) as exc_info:
        client._parse_json_content(json.dumps(wrong_types))
    err = str(exc_info.value)
    assert "invalid field types" in err
    assert "'evidence_summary' expected str, got int" in err
    assert "'counterevidence' expected str, got list" in err


def test_regression_8_provider_output_token_truncation():
    """Scenario 8: Token cutoff reported via finish_reason='length'."""
    client = NemotronClient(api_key="mock_key")
    truncated = '{"candidate_gap_statement": "The analyzed corpus shows a sparse intersection (sparsity=1.0) across 65 documents, with retrieved titles spanning housing justice, sustainability education, interdisci'
    
    with pytest.raises(ValueError) as exc_info:
        client._parse_json_content(truncated, finish_reason="length")
    err = str(exc_info.value)
    assert "finish_reason='length'" in err
    assert "truncated" in err.lower()

    # Case where JSON is closed but missing fields because of length cutoff
    partial_closed = '{"candidate_gap_statement": "Cutoff occurred"}'
    with pytest.raises(ValueError) as exc_info2:
        client._parse_json_content(partial_closed, finish_reason="length")
    err2 = str(exc_info2.value)
    assert "finish_reason='length'" in err2
    assert "Missing required fields" in err2


def test_regression_9_provider_error_responses():
    """Scenario 9: Comprehensive provider HTTP and network error responses."""
    client = NemotronClient(api_key="mock_secret_key_abc", provider="openrouter")

    # 401 Auth error
    with patch("requests.post") as mock_p:
        mock_p.return_value = MagicMock(status_code=401, json=lambda: {"error": {"message": "Invalid API key"}})
        with pytest.raises(PermissionError) as exc_401:
            client.post_chat_completion([{"role": "user", "content": "hi"}])
        assert "mock_secret_key_abc" not in str(exc_401.value)
        assert "HTTP 401" in str(exc_401.value)

    # 429 Rate limit
    with patch("requests.post") as mock_p:
        mock_p.return_value = MagicMock(status_code=429, json=lambda: {"error": {"message": "Rate limit exceeded"}})
        with pytest.raises(RuntimeError) as exc_429:
            client.post_chat_completion([{"role": "user", "content": "hi"}])
        assert "HTTP 429" in str(exc_429.value)

    # 503 Server error
    with patch("requests.post") as mock_p:
        mock_p.return_value = MagicMock(status_code=503, json=lambda: {"error": {"message": "OpenRouter overloaded"}})
        with pytest.raises(RuntimeError) as exc_503:
            client.post_chat_completion([{"role": "user", "content": "hi"}])
        assert "HTTP 503" in str(exc_503.value)


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_test_key", "LLM_PROVIDER": "openrouter"})
@patch("requests.post")
def test_regression_10_successful_explanation_with_source_references(mock_post):
    """Scenario 10: Complete end-to-end grounded explanation with mapped source references."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "gen-reg-10",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(VALID_EXPLANATION_DATA)
                },
                "finish_reason": "stop"
            }
        ]
    }
    mock_post.return_value = mock_resp

    client = NemotronClient()
    packet = {
        "candidate_id": "cand_reg_10",
        "source_metadata": [
            {"paper_id": "p_alpha", "title": "Alpha Research", "year": 2023, "chunk_id": "chunk_1", "source_field": "abstract"},
            {"paper_id": "p_beta", "title": "Beta Analysis", "year": 2024, "chunk_id": "chunk_2", "source_field": "conclusion"},
        ]
    }

    explanation = client.generate_grounded_explanation("System instructions", packet)
    assert explanation.candidate_id == "cand_reg_10"
    assert explanation.candidate_gap_statement == VALID_EXPLANATION_DATA["candidate_gap_statement"]
    assert explanation.evidence_summary == VALID_EXPLANATION_DATA["evidence_summary"]
    assert explanation.literature_context == VALID_EXPLANATION_DATA["literature_context"]
    assert explanation.limitation_evidence == VALID_EXPLANATION_DATA["limitation_evidence"]
    assert explanation.temporal_context == VALID_EXPLANATION_DATA["temporal_context"]
    assert explanation.counterevidence == VALID_EXPLANATION_DATA["counterevidence"]
    assert explanation.evidence_limitations == VALID_EXPLANATION_DATA["evidence_limitations"]
    assert explanation.suggested_direction == VALID_EXPLANATION_DATA["suggested_direction"]
    assert explanation.grounding_status == "Success"

    assert len(explanation.source_references) == 2
    assert explanation.source_references[0].paper_id == "p_alpha"
    assert explanation.source_references[0].title == "Alpha Research"
    assert explanation.source_references[0].year == 2023
    assert explanation.source_references[0].evidence_field == "abstract"
    assert explanation.source_references[1].paper_id == "p_beta"
    assert explanation.source_references[1].evidence_field == "conclusion"

