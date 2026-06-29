"""Judges: uniform interface over LLM (and mock) backends.

A *judge* is identified by a model string and exposes a single method::

    judge.generate(prompt_text, *, task, seed) -> str

which returns the judge's RAW text output. Parsing the raw text into a label
(e.g. ``YES-UNSAFE`` -> 1) happens in``extract.py``.

Two backends are provided:

``MockJudge``
    Fully offline, deterministic pseudo-random judge. It needs no API key and
    lets the whole pipeline + notebook run with zero external dependencies.
    Its answer is a stable function of ``(prompt_text, seed)``, so re-running is
    reproducible, while *different* seeds sometimes flip the answer (useful for
    flip-rate / steerability experiments).

``LiteLLMJudge``
    Thin wrapper around `litellm <https://github.com/BerriAI/litellm>`_'s
    ``completion`` call, giving access to OpenAI, Anthropic, Gemini, Together,
    local OpenAI-compatible servers, etc. ``litellm`` is imported lazily inside
    ``generate`` so merely importing this module never requires it.

Choosing a judge
----------------
Use the :func:`make_judge` factory::

    from jss.judges import make_judge

    # Offline, no keys:
    j = make_judge("mock")

    # Hosted models (litellm reads keys from standard env vars):
    j = make_judge("openai/gpt-5")             # needs OPENAI_API_KEY
    j = make_judge("anthropic/claude-4-5-sonnet")  # needs ANTHROPIC_API_KEY
    j = make_judge("gemini/gemini-2.5-pro")    # needs GEMINI_API_KEY

    raw = j.generate("Is this safe?", task="safety", seed=0)

API keys
--------
This module never touches API keys. ``litellm`` reads them from the standard
provider environment variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
``GEMINI_API_KEY``, ``TOGETHER_API_KEY``, ...). Set the appropriate variable in
your environment before constructing a hosted judge.

Local / self-hosted endpoints (Ollama, vLLM, ...)
-------------------------------------------------
Point an OpenAI-compatible local server by passing ``api_base`` and using an
``openai/`` model string::

    # vLLM / Ollama exposing an OpenAI-compatible API on localhost:
    j = make_judge(
        "openai/llama-3.1-8b-instruct",
        api_base="http://localhost:8000/v1",
    )

Most local servers ignore the key, but litellm may still require *some* value;
set ``OPENAI_API_KEY`` to any non-empty string (e.g. ``"sk-noop"``) if needed.
"""

from __future__ import annotations

import hashlib
import time

__all__ = ["MockJudge", "LiteLLMJudge", "make_judge"]


def _stable_hash_int(prompt_text: str, seed: int) -> int:
    """Return a stable, non-negative integer hash of ``(prompt_text, seed)``.
    """
    payload = f"{seed}\x00{prompt_text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big")


class MockJudge:
    """Deterministic, offline judge for running the pipeline without API keys.

    The answer is a pure function of ``(prompt_text, seed)``.
    """

    def __init__(self, model: str = "mock", **kwargs) -> None:
        # Extra kwargs are accepted (and ignored) so MockJudge is a drop-in
        # replacement for LiteLLMJudge in the factory, e.g. temperature=...
        self.model = model

    def generate(self, prompt_text: str, *, task: str, seed: int = 0) -> str:
        """Return a raw token string appropriate for ``task``.

        For ``task == "safety"`` returns either ``"YES-UNSAFE"`` or
        ``"NO-SAFE"``. For ``task == "classification"`` returns ``"A"`` or
        ``"B"``. The choice is deterministic in ``(prompt_text, seed)``.
        """
        # Hash including the seed so that different seeds can flip the answer,
        # while the same (prompt, seed) is always identical.
        h = _stable_hash_int(prompt_text, seed)
        bit = h & 1  # low bit -> 0/1

        if task == "safety":
            return "YES-UNSAFE" if bit else "NO-SAFE"
        if task == "classification":
            return "A" if bit else "B"
        raise ValueError(
            f"MockJudge.generate: unknown task {task!r} "
            "(expected 'safety' or 'classification')"
        )


class LiteLLMJudge:
    """Judge backed by ``litellm.completion`` (OpenAI/Anthropic/Gemini/local/...).

    Parameters
    ----------
    model:
        A litellm model string, e.g. ``"openai/gpt-5"``,
        ``"anthropic/claude-4-5-sonnet"``, ``"gemini/gemini-2.5-pro"``, or
        ``"openai/<name>"`` together with ``api_base`` for a local
        OpenAI-compatible endpoint.
    temperature:
        Sampling temperature (default ``0.0`` for deterministic judging).
    max_tokens:
        Maximum tokens to generate (default ``512``).
    api_base:
        Optional base URL for a local / self-hosted OpenAI-compatible server.
    **extra:
        Any additional keyword arguments are forwarded to
        ``litellm.completion`` (e.g. ``api_key``, ``timeout``, ...).
    """

    #: Number of attempts before giving up.
    MAX_RETRIES = 4

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_base: str | None = None,
        **extra,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base
        self.extra = extra
        self._params_checked = False  # warn-once guard for dropped params

    def _warn_unsupported_params(self, litellm) -> None:
        """Warn once if the provider doesn't support a param we're sending.

        ``litellm.drop_params`` silently drops provider-unsupported params (e.g.
        Gemini ignores ``seed``); we surface that here so the user knows the
        setting isn't being applied. Only the core sampling params are checked.
        """
        if self._params_checked:
            return
        self._params_checked = True
        try:
            supported = set(litellm.get_supported_openai_params(model=self.model) or [])
        except Exception:
            return  # provider/params unknown -> don't guess
        if not supported:
            return
        dropped = [p for p in ("temperature", "max_tokens", "seed") if p not in supported]
        if dropped:
            print(
                f"[LiteLLMJudge] model={self.model!r}: provider does not support "
                f"{dropped} — these are ignored (dropped by litellm)."
            )

    def generate(self, prompt_text: str, *, task: str, seed: int = 0) -> str:
        """Call the model and return the first choice's message content.

        ``task`` is part of the uniform interface but does not affect the call
        (the prompt already encodes the task). ``seed`` is accepted for
        interface uniformity and passed through to litellm when supported; if the
        provider doesn't support a core param (e.g. Gemini ignores ``seed``), a
        one-time warning is printed so the user knows it isn't applied.

        On transient failures the call is retried up to ``MAX_RETRIES`` times
        with short exponential backoff. If every attempt fails, a brief warning
        is printed and ``""`` is returned (which parses to NaN downstream).
        """
        # Import lazily so that importing this module does not require litellm.
        import litellm

        # Let litellm drop provider-unsupported params (e.g. Gemini does
        # not accept `seed`) instead of raising, but flag it to the user once.
        litellm.drop_params = True
        self._warn_unsupported_params(litellm)

        opt: dict = {}
        if self.api_base is not None:
            opt["api_base"] = self.api_base
        # Pass seed through; litellm forwards it to providers that support it
        # and harmlessly ignores it otherwise.
        if seed is not None:
            opt["seed"] = seed
        opt.update(self.extra)

        last_err: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **opt,
                )
                content = resp.choices[0].message.content
                return content if content is not None else ""
            except Exception as err:  # noqa: BLE001 - intentional broad retry
                last_err = err
                if attempt < self.MAX_RETRIES - 1:
                    # Exponential backoff (1s, 2s, 4s, ...) to ride out transient
                    # errors and brief rate-limit (429) windows.
                    time.sleep(1.0 * (2**attempt))

        print(
            f"[LiteLLMJudge] model={self.model!r} failed after "
            f"{self.MAX_RETRIES} attempts: {last_err!r}. Returning empty string."
        )
        return ""


def make_judge(model: str, **kwargs):
    """Factory: build a judge from a model string.

    ``"mock"`` (or any string starting with ``"mock"``) returns a
    :class:`MockJudge`; anything else returns a :class:`LiteLLMJudge`.

    Extra keyword arguments are forwarded to the judge constructor (e.g.
    ``temperature``, ``max_tokens``, ``api_base``).
    """
    if model == "mock" or model.startswith("mock"):
        return MockJudge(model, **kwargs)
    return LiteLLMJudge(model, **kwargs)
