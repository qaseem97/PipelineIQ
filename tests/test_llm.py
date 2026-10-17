"""GemmaClient retry/fallback behaviour, with the Gemini SDK call faked."""
import pytest

from pipelineiq import config
from pipelineiq.llm import GemmaClient, LLMError


class Resp:
    def __init__(self, text):
        self.text = text


def client_with(responses):
    """responses: list of Exception or str, consumed one per API call."""
    c = GemmaClient.__new__(GemmaClient)
    calls, retries = [], []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append({"model": model, "thinking": getattr(config, "thinking_config", None)})
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return Resp(r)

    c.client = type("C", (), {"models": Models()})()
    c.on_retry = retries.append
    c._sleep = lambda s: None
    return c, calls, retries


def test_recovers_from_a_500():
    c, calls, retries = client_with([RuntimeError("500 INTERNAL"), "ok"])
    assert c.generate("hi", model=config.MANAGER_MODEL, thinking=True) == "ok"
    assert len(calls) == 2 and len(retries) == 1


def test_switches_off_thinking_then_falls_back_to_other_gemma():
    err = RuntimeError("500 INTERNAL. Internal error encountered.")
    c, calls, retries = client_with([err, err, err, err, "from fallback"])
    assert c.generate("hi", model=config.MANAGER_MODEL, thinking=True) == "from fallback"
    assert [x["model"] for x in calls] == [config.MANAGER_MODEL] * 4 + [config.WORKER_MODEL]
    assert calls[0]["thinking"].thinking_level.value.lower() == "high"
    assert calls[2]["thinking"].thinking_level.value.lower() == "minimal"


def test_non_retryable_error_fails_fast():
    c, calls, _ = client_with([ValueError("400 INVALID_ARGUMENT: bad request")])
    with pytest.raises(LLMError):
        c.generate("hi")
    assert len(calls) == 1


def test_gives_up_after_all_attempts():
    c, calls, _ = client_with([RuntimeError("503 UNAVAILABLE")] * 6)
    with pytest.raises(LLMError, match="6 attempts"):
        c.generate("hi")
