import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
import requests

from app.schemas.rag import GroundedExplanation, SourceReference
from app.config import load_project_env, is_key_configured

# Ensure project .env is loaded from absolute path
load_project_env()

logger = logging.getLogger(__name__)

REQUIRED_EXPLANATION_FIELDS = [
    "candidate_gap_statement",
    "evidence_summary",
    "literature_context",
    "limitation_evidence",
    "temporal_context",
    "counterevidence",
    "evidence_limitations",
    "suggested_direction",
]

class NemotronClient:
    """
    Unified LLM Client supporting both OpenRouter and direct NVIDIA API endpoints.
    Defaults to OpenRouter provider with 'nvidia/nemotron-3-ultra-550b-a55b:free'
    while maintaining full backward compatibility with legacy NVIDIA Nemotron settings.
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self._explicit_key = api_key
        # Ensure latest env is loaded from absolute path
        load_project_env()

        # Resolve provider: openrouter or nvidia
        raw_provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or ("openrouter" if os.getenv("OPENROUTER_API_KEY") else "nvidia")
        )
        self.provider = raw_provider.strip().lower()

        if self.provider == "openrouter":
            raw_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
            self.api_key = raw_key.strip("'\" \t\r\n") if raw_key else ""
            self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
            self.model = model or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        else:
            self.provider = "nvidia"
            raw_key = api_key if api_key is not None else os.getenv("NVIDIA_NEMOTRON_API_KEY")
            self.api_key = raw_key.strip("'\" \t\r\n") if raw_key else ""
            self.base_url = (base_url or os.getenv("NVIDIA_NEMOTRON_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
            self.model = model or os.getenv("NVIDIA_NEMOTRON_MODEL", "nvidia/nemotron-4-340b-instruct")

        # Configurable request timeout (defaults to 60s)
        env_timeout = os.getenv("LLM_TIMEOUT") or os.getenv("OPENROUTER_TIMEOUT") or "60"
        try:
            self.timeout = int(timeout if timeout is not None else env_timeout)
        except (ValueError, TypeError):
            self.timeout = 60

        # Configurable output max tokens (defaults to 2500 to prevent truncated JSON)
        env_max_tokens = os.getenv("LLM_MAX_TOKENS") or os.getenv("OPENROUTER_MAX_TOKENS") or "2500"
        try:
            self.max_tokens = int(env_max_tokens)
        except (ValueError, TypeError):
            self.max_tokens = 2500

        # Optional OpenRouter reasoning parameters
        self.reasoning_enabled = os.getenv("OPENROUTER_REASONING_ENABLED", "").lower() in ("true", "1", "yes")
        self.reasoning_effort = os.getenv("OPENROUTER_REASONING_EFFORT")
        self.reasoning_max_tokens = os.getenv("OPENROUTER_REASONING_MAX_TOKENS")

    def is_configured(self) -> bool:
        """
        Returns True if a real API key is configured and not a placeholder.
        Dynamically reloads .env if not currently configured to pick up runtime edits.
        """
        if is_key_configured(self.api_key):
            return True

        # Check if environment already has a configured key
        key_name = "OPENROUTER_API_KEY" if self.provider == "openrouter" else "NVIDIA_NEMOTRON_API_KEY"
        current_env_key = os.getenv(key_name) or ""
        clean_env_key = current_env_key.strip("'\" \t\r\n")
        if is_key_configured(clean_env_key):
            self.api_key = clean_env_key
            return True

        # If key was not passed explicitly and env var is not configured, reload .env from disk
        if self._explicit_key is None:
            load_project_env(override=True)
            fresh_key = os.getenv(key_name) or ""
            fresh_clean = fresh_key.strip("'\" \t\r\n")
            if is_key_configured(fresh_clean):
                self.api_key = fresh_clean
                return True

        return False

    def post_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a POST request to /chat/completions with structured error handling.
        Returns a dict containing 'content', 'model', 'id', 'usage', and optional 'reasoning_details'.
        """
        if not self.is_configured():
            key_name = "OPENROUTER_API_KEY" if self.provider == "openrouter" else "NVIDIA_NEMOTRON_API_KEY"
            raise ValueError(f"{key_name} is not configured. Please set {key_name} in backend/.env or your environment.")

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/researchgapx"
            headers["X-Title"] = "ResearchGap-X"

        if extra_headers:
            headers.update(extra_headers)

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        # Optional reasoning configuration for OpenRouter
        if self.provider == "openrouter":
            reasoning_config: Dict[str, Any] = {}
            if self.reasoning_effort:
                reasoning_config["effort"] = self.reasoning_effort
            elif self.reasoning_max_tokens:
                try:
                    reasoning_config["max_tokens"] = int(self.reasoning_max_tokens)
                except (ValueError, TypeError):
                    pass
            elif self.reasoning_enabled:
                reasoning_config["enabled"] = True

            if reasoning_config:
                payload["reasoning"] = reasoning_config

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {self.provider} ({self.model}) timed out after {self.timeout}s.")
        except requests.exceptions.ConnectionError as ce:
            raise ConnectionError(f"Failed to connect to {self.provider} at {self.base_url}: {str(ce)}")
        except Exception as e:
            raise RuntimeError(f"Network error during {self.provider} request: {str(e)}")

        # Distinguish error responses
        status_code = response.status_code
        if status_code != 200:
            err_detail = "Unknown error"
            try:
                err_json = response.json()
                if "error" in err_json:
                    err_obj = err_json["error"]
                    err_detail = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                else:
                    err_detail = str(err_json)
            except Exception:
                err_detail = response.text[:200] if response.text else f"HTTP {status_code}"

            if status_code in (401, 403):
                raise PermissionError(
                    f"{self.provider} authentication failed (HTTP {status_code}): Invalid or unauthorized API key."
                )
            elif status_code == 402:
                raise RuntimeError(
                    f"{self.provider} error (HTTP 402): Insufficient credits or payment required for model '{self.model}'."
                )
            elif status_code == 404:
                raise LookupError(
                    f"{self.provider} model '{self.model}' not found or unavailable (HTTP 404): {err_detail}"
                )
            elif status_code == 429:
                raise RuntimeError(
                    f"{self.provider} rate limit exceeded (HTTP 429). Detail: {err_detail}"
                )
            elif 500 <= status_code < 600:
                raise RuntimeError(
                    f"{self.provider} server error (HTTP {status_code}): {err_detail}"
                )
            else:
                raise RuntimeError(
                    f"{self.provider} API request failed with HTTP {status_code}: {err_detail}"
                )

        try:
            result = response.json()
        except Exception as e:
            raise ValueError(f"Failed to decode JSON response from {self.provider}: {str(e)}")

        if "choices" not in result or not isinstance(result["choices"], list):
            raise ValueError(f"Response from {self.provider} missing 'choices': {result}")
        choices = result["choices"]
        if len(choices) == 0:
            raise ValueError(f"Response from {self.provider} returned empty 'choices' list")

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice or not isinstance(first_choice["message"], dict):
            raise ValueError(f"Response from {self.provider} missing 'message' in choice")

        message = first_choice["message"]
        content = message.get("content") or ""

        # Preserve reasoning details internally if returned, without exposing to regular content
        reasoning_details = message.get("reasoning_details") or message.get("reasoning")

        if not content.strip() and not reasoning_details:
            raise ValueError(f"Response from {self.provider} returned empty message content")

        return {
            "content": content,
            "model": result.get("model", self.model),
            "id": result.get("id"),
            "usage": result.get("usage", {}),
            "finish_reason": first_choice.get("finish_reason"),
            "reasoning_details": reasoning_details,
            "raw": result,
        }

    def _parse_json_content(
        self,
        content: str,
        finish_reason: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Safely parses and validates JSON from LLM output.
        - Handles plain JSON, markdown code blocks, and harmless surrounding text.
        - Robustly catches token truncation (finish_reason='length' or unclosed braces).
        - Validates all REQUIRED_EXPLANATION_FIELDS and asserts their string type.
        - Retains safe diagnostic metadata without exposing raw sensitive responses.
        """
        if not content or not content.strip():
            diag = f"finish_reason='{finish_reason}', model='{model or self.model}'" if finish_reason else f"model='{model or self.model}'"
            raise ValueError(f"LLM returned empty message content ({diag}).")

        cleaned = content.strip()
        candidate_json: Optional[str] = None

        # 1. Match markdown code fences ```json { ... } ``` or ``` { ... } ```
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            candidate_json = fence_match.group(1).strip()
        else:
            # Strip unclosed opening markdown fence if output was truncated mid-fence
            if cleaned.startswith("```json"):
                cleaned_no_fence = cleaned[7:].strip()
            elif cleaned.startswith("```"):
                cleaned_no_fence = cleaned[3:].strip()
            else:
                cleaned_no_fence = cleaned

            # 2. Extract outermost JSON object bounds: first '{' and last '}'
            start = cleaned_no_fence.find("{")
            end = cleaned_no_fence.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate_json = cleaned_no_fence[start : end + 1].strip()
            elif start != -1 and (end == -1 or end <= start):
                if finish_reason == "length":
                    raise ValueError(
                        f"LLM response was truncated due to output token limit (finish_reason='length', length={len(content)} chars). Opening '{{' without closing '}}'."
                    )
                raise ValueError(
                    f"Incomplete JSON in LLM response: opening '{{' found at pos {start} but no closing '}}' (length={len(content)} chars)."
                )
            else:
                if finish_reason == "length":
                    raise ValueError(
                        f"LLM response was truncated due to output token limit (finish_reason='length', length={len(content)} chars). No JSON object found."
                    )
                raise ValueError(
                    f"Could not parse valid JSON from LLM response (length={len(content)} chars, model='{model or self.model}'). No JSON object found."
                )

        parsed: Optional[Any] = None
        parse_error_pos: Optional[int] = None
        parse_error_msg: Optional[str] = None

        # 3. Attempt parsing with strict=False (allows unescaped control chars like literal newlines)
        try:
            parsed = json.loads(candidate_json, strict=False)
        except json.JSONDecodeError as jde:
            parse_error_pos = jde.pos
            parse_error_msg = jde.msg
            # Fallback: remove harmless trailing commas before closing braces/brackets
            sanitized = re.sub(r',\s*([\]\}])', r'\1', candidate_json)
            try:
                parsed = json.loads(sanitized, strict=False)
            except json.JSONDecodeError:
                pass

        if parsed is None:
            if finish_reason == "length":
                raise ValueError(
                    f"LLM response was truncated due to output token limit (finish_reason='length', length={len(content)} chars, parse error at pos {parse_error_pos}: {parse_error_msg})."
                )
            raise ValueError(
                f"Could not parse valid JSON from LLM response (length={len(content)} chars, error at pos {parse_error_pos}: {parse_error_msg})."
            )

        # 4. Validate parsed structure
        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM response parsed as {type(parsed).__name__}, expected a JSON object (dict)."
            )

        # 5. Validate required fields presence
        missing_fields = [f for f in REQUIRED_EXPLANATION_FIELDS if f not in parsed]
        if missing_fields:
            if finish_reason == "length":
                raise ValueError(
                    f"LLM response was truncated (finish_reason='length'). Missing required fields: {', '.join(missing_fields)}."
                )
            raise ValueError(
                f"LLM response missing required fields: {', '.join(missing_fields)}."
            )

        # 6. Validate field types (must be str)
        invalid_types = [
            f for f in REQUIRED_EXPLANATION_FIELDS
            if not isinstance(parsed[f], str)
        ]
        if invalid_types:
            details = [f"'{f}' expected str, got {type(parsed[f]).__name__}" for f in invalid_types]
            raise ValueError(
                f"LLM response contains invalid field types: {', '.join(details)}."
            )

        return parsed

    def generate_grounded_explanation(
        self,
        system_prompt: str,
        evidence_packet: Dict[str, Any],
    ) -> GroundedExplanation:
        """
        Generates a structured GroundedExplanation from an evidence packet.
        Preserves complete backward compatibility with existing RAG routes.
        Uses configured max_tokens and enables response_format for structured output.
        """
        prompt = (
            f"{system_prompt}\n\n"
            f"Evidence Packet:\n{json.dumps(evidence_packet, indent=2)}\n\n"
            f"Please respond with valid JSON only matching the schema."
        )

        try:
            extra_payload: Dict[str, Any] = {}
            if self.provider == "openrouter":
                extra_payload["response_format"] = {"type": "json_object"}

            completion = self.post_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=self.max_tokens,
                **extra_payload,
            )
            content = completion.get("content", "")
            finish_reason = completion.get("finish_reason")
            model_used = completion.get("model", self.model)

            parsed = self._parse_json_content(
                content=content,
                finish_reason=finish_reason,
                model=model_used,
            )

            # Map source references back from evidence packet metadata
            source_refs: List[SourceReference] = []
            for meta in evidence_packet.get("source_metadata", []):
                source_refs.append(SourceReference(
                    paper_id=meta.get("paper_id", ""),
                    title=meta.get("title", ""),
                    year=meta.get("year", 0),
                    chunk_id=meta.get("chunk_id"),
                    evidence_field=meta.get("source_field", "abstract"),
                    similarity_score=None
                ))

            return GroundedExplanation(
                candidate_id=evidence_packet.get("candidate_id", ""),
                candidate_gap_statement=parsed["candidate_gap_statement"].strip(),
                evidence_summary=parsed["evidence_summary"].strip(),
                literature_context=parsed["literature_context"].strip(),
                limitation_evidence=parsed["limitation_evidence"].strip(),
                temporal_context=parsed["temporal_context"].strip(),
                counterevidence=parsed["counterevidence"].strip(),
                evidence_limitations=parsed["evidence_limitations"].strip(),
                suggested_direction=parsed["suggested_direction"].strip(),
                source_references=source_refs,
                grounding_status="Success"
            )
        except (ValueError, PermissionError, LookupError, TimeoutError, ConnectionError):
            raise
        except Exception as e:
            raise RuntimeError(f"Nemotron API call failed: {str(e)}")
