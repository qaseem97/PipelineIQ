"""Thin wrapper around Gemma 4 on the Gemini API, plus an offline fake for tests."""
from __future__ import annotations

import itertools
import json
import re
import time
from typing import Callable

from . import config


class LLMError(Exception):
    pass


def extract_json(text: str) -> dict | list:
    """Pull the first JSON object/array out of a model reply (handles ```json fences and chatter)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = min([i for i in (candidate.find("{"), candidate.find("[")) if i != -1], default=-1)
    if start == -1:
        raise LLMError(f"No JSON found in model reply: {text[:200]!r}")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(candidate[start:])
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e}") from e
    return obj


class GemmaClient:
    """Calls Gemma 4 through the google-genai SDK."""

    # Waits between attempts (seconds). Server errors (500/503) and rate limits (429)
    # on the Gemini API are usually temporary, so back off and try again.
    BACKOFF = (2, 5, 10, 20, 30)

    def __init__(self, api_key: str | None = None, on_retry=None):
        from google import genai  # imported lazily so tests run without the SDK configured

        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.on_retry = on_retry or (lambda info: None)
        self._sleep = time.sleep

    @staticmethod
    def _fallback(model: str) -> str:
        """The other Gemma 4 model, used when one keeps failing."""
        return config.MANAGER_MODEL if model == config.WORKER_MODEL else config.WORKER_MODEL

    @staticmethod
    def _retryable(err: Exception) -> bool:
        s = str(err).lower()
        return any(k in s for k in ("500", "502", "503", "504", "429", "internal", "unavailable",
                                    "overloaded", "resource_exhausted", "deadline", "timeout",
                                    "empty response", "connection"))

    def _call(self, model, contents, cfg, thinking_cfg):
        from google.genai import types
        try:
            config_obj = types.GenerateContentConfig(**cfg, thinking_config=thinking_cfg) if thinking_cfg \
                else types.GenerateContentConfig(**cfg)
            resp = self.client.models.generate_content(model=model, contents=contents, config=config_obj)
        except Exception as e:  # some models reject thinking settings; retry without them
            if thinking_cfg is None or "thinking" not in str(e).lower():
                raise
            resp = self.client.models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
        if not resp.text:
            raise LLMError("Empty response from model")
        return resp.text

    def generate(self, prompt: str, *, system: str | None = None, model: str | None = None,
                 images: list[bytes] | None = None, thinking: bool = False,
                 temperature: float = 0.4) -> str:
        from google.genai import types

        contents: list = []
        for img in images or []:
            mime = "image/png" if img[:8] == b"\x89PNG\r\n\x1a\n" else (
                "image/webp" if img[8:12] == b"WEBP" else "image/jpeg")
            contents.append(types.Part.from_bytes(data=img, mime_type=mime))
        contents.append(prompt)

        cfg = dict(system_instruction=system, temperature=temperature)
        primary = model or config.WORKER_MODEL
        # Attempt plan: primary model as asked -> primary without thinking -> other Gemma 4 model.
        plan = [(primary, thinking), (primary, thinking), (primary, False), (primary, False),
                (self._fallback(primary), False), (self._fallback(primary), False)]

        errors = []
        for i, (m, think) in enumerate(plan):
            thinking_cfg = types.ThinkingConfig(thinking_level="high" if think else "minimal")
            try:
                return self._call(m, contents, cfg, thinking_cfg)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{m}: {e}")
                if not self._retryable(e) or i == len(plan) - 1:
                    break
                wait = self.BACKOFF[min(i, len(self.BACKOFF) - 1)]
                nxt = plan[i + 1]
                self.on_retry({"model": m, "error": str(e)[:200], "wait": wait,
                               "next_model": nxt[0], "thinking": nxt[1]})
                self._sleep(wait)
        raise LLMError(f"Gemma call failed after {len(errors)} attempts. Last error: {errors[-1]}")

    def generate_json(self, prompt: str, **kwargs) -> dict | list:
        text = self.generate(prompt + "\n\nRespond with valid JSON only.", **kwargs)
        return extract_json(text)


class FakeLLM:
    """Offline stand-in. `responder(prompt, system)` returns the text the 'model' replies with."""

    def __init__(self, responder: Callable[[str, str | None], str] | None = None):
        self.responder = responder or (lambda prompt, system: '{"ok": true}')
        self.calls: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "system": system, **kwargs})
        return self.responder(prompt, system)

    def generate_json(self, prompt: str, **kwargs) -> dict | list:
        return extract_json(self.generate(prompt, **kwargs))


class TrackedLLM:
    """Wraps any LLM client and reports each model call to `on_call(kind, data)`:
    'llm_start' when a request goes out, 'llm_end' with its duration and outcome when it returns."""

    _ids = itertools.count(1)  # global, so call ids stay unique across runs and resumes

    def __init__(self, inner, on_call: Callable[[str, dict], None]):
        self.inner = inner
        self.on_call = on_call

    def generate(self, prompt: str, *, model: str | None = None, **kwargs) -> str:
        call = {"call": next(self._ids), "model": model or config.WORKER_MODEL}
        self.on_call("llm_start", call)
        t0 = time.monotonic()
        try:
            text = self.inner.generate(prompt, model=model, **kwargs)
        except Exception as e:
            self.on_call("llm_end", {**call, "ok": False, "ms": round((time.monotonic() - t0) * 1000),
                                     "error": str(e)[:200]})
            raise
        self.on_call("llm_end", {**call, "ok": True, "ms": round((time.monotonic() - t0) * 1000)})
        return text

    def generate_json(self, prompt: str, **kwargs) -> dict | list:
        return extract_json(self.generate(prompt + "\n\nRespond with valid JSON only.", **kwargs))


def get_llm(on_retry=None):
    return FakeLLM() if config.USE_FAKE_LLM else GemmaClient(on_retry=on_retry)
