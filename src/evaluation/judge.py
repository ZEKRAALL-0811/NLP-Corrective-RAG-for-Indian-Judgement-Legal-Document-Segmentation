"""
Module 10: DeepEval LLM Judge with Token-Aware Rate Limiting & Gemini/Gemma Provider (src/evaluation/judge.py)

Implements:
1. TokenAwareRateLimiter:
   - Sliding 60-second window enforcing at most 14,000 tokens/min and 25 requests/min.
   - Pre-call token estimation: len(prompt) / 3.5.
   - Post-call correction from API usage metadata.
   - Strict sequential execution with threading lock.
   - Exponential backoff and Retry-After handling.
   - Daily request budget enforcement (JUDGE_DAILY_REQUEST_BUDGET).
   - Full ledger logging (including 429s) to artifacts/logs/judge_usage.jsonl.
2. GeminiJudge (DeepEvalBaseLLM):
   - Primary: Gemma 4 31B (gemma-4-31b-it)
   - Secondary: Gemma 4 26B (gemma-4-26b-a4b-it)
   - Cross-check: Gemini 3.1 Flash-Lite (gemini-3.1-flash-lite)
   - Native JSON mode (response_mime_type="application/json") and system instructions.
   - Robust Pydantic schema validation with one repair retry.
   - No silent fallback (errors halt execution and report).
3. GroqJudge (DeepEvalBaseLLM):
   - Legacy provider, ONLY accessible when JUDGE_PROVIDER="groq".
"""

import os
import json
import time
import threading
import hashlib
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from collections import deque
from pydantic import BaseModel
from dotenv import load_dotenv

from deepeval.models import DeepEvalBaseLLM
import src.config as cfg

load_dotenv()

USAGE_LEDGER_PATH = cfg.ARTIFACTS_DIR / "logs" / "judge_usage.jsonl"
CACHE_PATH = cfg.CACHE_DIR / "judge_cache.json"


class BudgetExceededError(Exception):
    """Raised when the daily judge request or token budget is reached."""
    pass


class TokenAwareRateLimiter:
    """
    Sliding 60-second window rate limiter for LLM judges.
    Limits:
      - Max 14,000 tokens per minute
      - Max 25 requests per minute
      - Daily request budget: Flash-Lite 450, Gemma 26B/31B 14,400
    """

    MODEL_CAPS = {
        "gemini-3.5-flash-lite": cfg.JUDGE_DAILY_REQUEST_BUDGET,  # 450
        "gemma-4-26b-a4b-it": cfg.GEMMA_RPD_LIMIT,              # 14400
        "gemma-4-31b-it": cfg.GEMMA_RPD_LIMIT,                  # 14400
    }

    def __init__(
        self,
        max_tpm: int = cfg.RATE_LIMIT_MAX_TPM,
        max_rpm: int = cfg.RATE_LIMIT_MAX_RPM,
        daily_budget: Optional[int] = None,
    ):
        self.max_tpm = max_tpm
        self.max_rpm = max_rpm
        self.daily_budget = daily_budget
        self.window_seconds = 60.0
        self.lock = threading.Lock()
        # Records: list of (timestamp, token_count)
        self.history: deque = deque()

    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate prompt tokens before sending (characters / 3.5)."""
        return max(1, int(len(prompt) / 3.5))

    def _prune_old_entries(self, now: float) -> None:
        """Remove entries older than sliding window."""
        cutoff = now - self.window_seconds
        while self.history and self.history[0][0] <= cutoff:
            self.history.popleft()

    def get_today_request_count(self, model: Optional[str] = None) -> int:
        """Count successful/failed calls logged today for a specific model."""
        if not USAGE_LEDGER_PATH.exists():
            return 0
        today_prefix = time.strftime("%Y-%m-%d")
        count = 0
        try:
            with open(USAGE_LEDGER_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("timestamp", "").startswith(today_prefix):
                            if model is None or entry.get("model") == model:
                                count += 1
        except Exception:
            pass
        return count

    def acquire(self, prompt: str, model: str = "gemini-3.5-flash-lite") -> Tuple[int, float]:
        """
        Blocks until rate limit budget allows the request.
        Returns (estimated_tokens, wait_time_seconds).
        """
        est_tokens = self.estimate_prompt_tokens(prompt)
        start_wait = time.time()
        model_cap = self.MODEL_CAPS.get(model, cfg.JUDGE_DAILY_REQUEST_BUDGET)
        effective_cap = min(self.daily_budget, model_cap) if self.daily_budget is not None else model_cap

        while True:
            with self.lock:
                now = time.time()
                self._prune_old_entries(now)

                # Check daily budget for this model
                today_reqs = self.get_today_request_count(model=model)
                if today_reqs >= effective_cap:
                    raise BudgetExceededError(
                        f"Daily request budget (JUDGE_DAILY_REQUEST_BUDGET={effective_cap}) reached for model {model} today ({today_reqs} requests). "
                        "Stopping cleanly."
                    )

                curr_rpm = len(self.history)
                curr_tpm = sum(entry[1] for entry in self.history)

                # Check if this request fits in 60s sliding window
                if (curr_rpm + 1 <= self.max_rpm) and (curr_tpm + est_tokens <= self.max_tpm):
                    # Reserve tentative slot in window
                    self.history.append((now, est_tokens))
                    total_waited = time.time() - start_wait
                    return est_tokens, total_waited

                # If limits breached, calculate wait time until oldest request rolls off
                oldest_ts = self.history[0][0]
                sleep_time = max(0.2, self.window_seconds - (now - oldest_ts) + 0.1)

            time.sleep(sleep_time)

    def correct_usage(self, est_tokens: int, actual_tokens: int) -> None:
        """Correct estimated tokens in the sliding window with actual metadata tokens."""
        with self.lock:
            # Find the most recent matching entry with est_tokens and adjust
            for i in reversed(range(len(self.history))):
                ts, t_count = self.history[i]
                if t_count == est_tokens:
                    self.history[i] = (ts, actual_tokens)
                    break

    def log_call(
        self,
        provider: str,
        model: str,
        status: int,
        prompt_tokens: int,
        completion_tokens: int,
        wait_time_s: float,
        metric: str = "general",
    ) -> None:
        """Appends call details (including 429s) to artifacts/logs/judge_usage.jsonl."""
        USAGE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        p_tok = prompt_tokens if prompt_tokens is not None else 0
        c_tok = completion_tokens if completion_tokens is not None else 0
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model": model,
            "metric": metric,
            "status": status,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok,
            "wait_time_s": round(wait_time_s, 3),
        }
        with open(USAGE_LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()


# Global Singleton Limiter
RATE_LIMITER = TokenAwareRateLimiter()


class GeminiJudge(DeepEvalBaseLLM):
    """
    DeepEval LLM Judge interfacing with Google Gemini / Gemma models via google-genai SDK.
    Strictly adheres to sliding window token rate limits, JSON schema validation,
    repair retry, and sequential execution.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        metric: Optional[str] = None,
        max_attempts: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

        self.selected_model = model_name or cfg.JUDGE_MODEL_PRIMARY
        self.metric = metric or "general"
        default_attempts = 4 if self.selected_model.startswith("gemma") else 6
        self.max_attempts = max_attempts if max_attempts is not None else default_attempts
        self.cache: Dict[str, Any] = self._load_cache()

        from google import genai
        from google.genai import types
        timeout_ms = 120000 if "gemma" in self.selected_model else 45000
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        super().__init__(model_name=self.selected_model, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        return f"gemini-client:{self.selected_model}"

    def get_model_name(self, *args, **kwargs) -> str:
        return self.selected_model

    def _load_cache(self) -> Dict[str, Any]:
        if CACHE_PATH.exists():
            try:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def _parse_schema(self, content: str, schema: BaseModel) -> Optional[BaseModel]:
        s = content.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

        try:
            return schema.model_validate(json.loads(s))
        except Exception:
            # Fallback regex search for JSON object
            b0 = s.find("{")
            b1 = s.rfind("}")
            if b0 != -1 and b1 != -1 and b1 > b0:
                try:
                    return schema.model_validate(json.loads(s[b0 : b1 + 1]))
                except Exception:
                    pass
        return None

    def _call_api_with_retry(
        self,
        prompt: str,
        system_instruction: str,
        is_repair: bool = False,
    ) -> Tuple[str, int, int]:
        from google.genai import types

        max_attempts = getattr(self, "max_attempts", 4 if self.selected_model.startswith("gemma") else 6)
        backoff_delay = 2.0

        for attempt in range(max_attempts):
            est_tokens, wait_time = RATE_LIMITER.acquire(prompt, model=self.selected_model)

            sanitized_prompt = prompt.replace("\ufffd", " - ").replace("otherwise I WILL DIE.", "").replace("otherwise I will die.", "")
            sanitized_sys = system_instruction.replace("\ufffd", " - ")

            safety_settings = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]

            try:
                if self.selected_model.startswith("gemma"):
                    # Prompt-instructed JSON mode for Gemma: embed system instruction directly in contents, omit system_instruction in config
                    sanitized_prompt_payload = f"{sanitized_sys}\n\n{sanitized_prompt}\n\nIMPORTANT: Respond strictly in valid JSON adhering to the schema. Output valid JSON only."
                    config = types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=2048,
                        safety_settings=safety_settings,
                    )
                else:
                    sanitized_prompt_payload = sanitized_prompt
                    config = types.GenerateContentConfig(
                        system_instruction=sanitized_sys,
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=2048,
                        safety_settings=safety_settings,
                    )

                resp = self.client.models.generate_content(
                    model=self.selected_model,
                    contents=sanitized_prompt_payload,
                    config=config,
                )

                # Extract tokens
                usage = getattr(resp, "usage_metadata", None)
                p_raw = getattr(usage, "prompt_token_count", None) if usage else None
                c_raw = getattr(usage, "candidates_token_count", None) if usage else None
                tot_raw = getattr(usage, "total_token_count", None) if usage else None

                p_tokens = p_raw if p_raw is not None else est_tokens
                c_tokens = c_raw if c_raw is not None else 0
                tot_tokens = tot_raw if tot_raw is not None else (p_tokens + c_tokens)

                RATE_LIMITER.correct_usage(est_tokens, tot_tokens)
                RATE_LIMITER.log_call(
                    provider="gemini",
                    model=self.selected_model,
                    status=200,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    wait_time_s=wait_time,
                    metric=self.metric,
                )

                text_out = resp.text or ""
                return text_out, p_tokens, c_tokens

            except Exception as e:
                err_str = str(e)
                # Check for 429 / 500 / 504
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
                is_504 = "504" in err_str or "GATEWAY_TIMEOUT" in err_str or "timed out" in err_str.lower()
                is_500 = "500" in err_str or "INTERNAL" in err_str
                status_code = 429 if is_429 else (504 if is_504 else (500 if is_500 else 500))
                print(f"[GeminiJudge API Error] status={status_code}: {err_str}", flush=True)

                RATE_LIMITER.log_call(
                    provider="gemini",
                    model=self.selected_model,
                    status=status_code,
                    prompt_tokens=est_tokens,
                    completion_tokens=0,
                    wait_time_s=wait_time,
                    metric=self.metric,
                )

                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Gemini API call failed after {max_attempts} attempts: {err_str}")

                if is_429:
                    sleep_s = backoff_delay * (2 ** attempt) + 1.0
                    print(f"[GeminiJudge] 429 rate limit hit for {self.selected_model}. Backing off {sleep_s:.1f}s (attempt {attempt+1}/{max_attempts})...", flush=True)
                    time.sleep(sleep_s)
                    continue

                if is_504:
                    sleep_s = 5.0 * (2 ** attempt) + 2.0
                    print(f"[GeminiJudge] 504 Gateway Timeout for {self.selected_model}. Backing off {sleep_s:.1f}s (attempt {attempt+1}/{max_attempts})...", flush=True)
                    time.sleep(sleep_s)
                    continue

                is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower()
                if is_503:
                    sleep_s = 4.0 * (2 ** attempt) + 2.0
                    print(f"[GeminiJudge] 503 high demand spike for {self.selected_model}. Backing off {sleep_s:.1f}s (attempt {attempt+1}/{max_attempts})...", flush=True)
                    time.sleep(sleep_s)
                    continue

                if "404" in err_str or "NOT_FOUND" in err_str:
                    # Rule R3: no silent fallback. Stop and report.
                    raise RuntimeError(f"Configured model {self.selected_model} returned 404 NOT FOUND: {err_str}")

                time.sleep(backoff_delay * (2 ** attempt))

        raise RuntimeError(f"Gemini API failed to generate content for {self.selected_model}.")

    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args,
        **kwargs,
    ) -> Union[str, BaseModel]:
        metric_name = self.metric
        schema_name = getattr(schema, "__name__", "")
        cache_key = hashlib.sha256(
            f"{metric_name}:{self.selected_model}:{prompt}:{schema_name}".encode("utf-8")
        ).hexdigest()


        if cache_key in self.cache:
            raw = self.cache[cache_key]["content"]
            if schema:
                parsed = self._parse_schema(raw, schema)
                if parsed is not None:
                    return parsed
            return raw

        sys_prompt = (
            "You are a strict, precise legal evaluation judge. "
            "Evaluate the user's test case strictly adhering to the JSON schema. "
            "Output ONLY valid JSON."
        )

        content, p_tok, c_tok = self._call_api_with_retry(prompt=prompt, system_instruction=sys_prompt)

        # Schema parsing & 1 repair retry
        if schema:
            instance = self._parse_schema(content, schema)
            if instance is None:
                # 1 repair retry
                repair_prompt = (
                    f"The previous output failed JSON validation for schema {schema_name}.\n"
                    f"Previous Output:\n{content}\n\n"
                    f"Please re-evaluate and return strictly valid JSON matching {schema_name}:\n{prompt}"
                )
                repaired_content, _, _ = self._call_api_with_retry(
                    prompt=repair_prompt,
                    system_instruction=sys_prompt,
                    is_repair=True,
                )
                instance = self._parse_schema(repaired_content, schema)
                if instance is None:
                    raise ValueError(f"Failed to parse JSON response into schema {schema_name} after repair retry.")
                content = repaired_content

            self.cache[cache_key] = {"content": content, "prompt_tokens": p_tok, "completion_tokens": c_tok}
            self._save_cache()
            return instance

        self.cache[cache_key] = {"content": content, "prompt_tokens": p_tok, "completion_tokens": c_tok}
        self._save_cache()
        return content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args,
        **kwargs,
    ) -> Union[str, BaseModel]:
        return self.generate(prompt=prompt, schema=schema, *args, **kwargs)


class GroqJudge(DeepEvalBaseLLM):
    """
    Legacy Groq Judge.
    Per Rule R3: The Groq key must not be used unless JUDGE_PROVIDER=groq.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        metric: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if cfg.JUDGE_PROVIDER != "groq":
            raise PermissionError("Groq key must not be used unless JUDGE_PROVIDER=groq.")

        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        self.selected_model = model_name or cfg.GROQ_JUDGE_MODEL
        self.metric = metric or "general"
        super().__init__(model_name=self.selected_model, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        return "groq-http-client"

    def get_model_name(self, *args, **kwargs) -> str:
        return self.selected_model

    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args,
        **kwargs,
    ) -> Union[str, BaseModel]:
        raise NotImplementedError("Groq provider disabled. Use GeminiJudge with JUDGE_PROVIDER=gemini.")

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        *args,
        **kwargs,
    ) -> Union[str, BaseModel]:
        return self.generate(prompt=prompt, schema=schema, *args, **kwargs)


def get_judge(
    model_name: Optional[str] = None,
    crosscheck: bool = False,
    metric: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> DeepEvalBaseLLM:
    """Factory creating judge instance according to configured JUDGE_PROVIDER."""
    provider = getattr(cfg, "JUDGE_PROVIDER", "gemini").lower()
    if provider == "groq":
        return GroqJudge(model_name=model_name, metric=metric)

    # Gemini provider
    if crosscheck:
        chosen_model = model_name or cfg.JUDGE_CROSSCHECK_MODEL
    else:
        chosen_model = model_name or cfg.JUDGE_MODEL_PRIMARY

    return GeminiJudge(model_name=chosen_model, metric=metric, max_attempts=max_attempts)
