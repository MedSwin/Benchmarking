from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.config import settings
from app.metrics import norm_text
from app.models import MODEL_PROVIDER, ProviderResponse, TargetModel
from app.rate import AsyncRateLimiter

logger = logging.getLogger(__name__)


class ProviderPool:
    def __init__(self) -> None:
        self._key_cycles: Dict[str, Any] = {}
        self._limiters: Dict[str, List[AsyncRateLimiter]] = {}
        self._lock = asyncio.Lock()
        for provider, keys in settings.provider_keys.items():
            if keys:
                self._key_cycles[provider] = itertools.cycle(enumerate(keys))
                rpm = getattr(settings, f"{provider}_requests_per_minute")
                backoff_factor = settings.rate_limit_backoff_factor
                recovery_seconds = settings.rate_limit_recovery_seconds
                self._limiters[provider] = [
                    AsyncRateLimiter(rpm, backoff_factor, recovery_seconds) for _ in keys
                ]
        self._model_cycles: Dict[str, Any] = {}
        self._model_options: Dict[str, List[str]] = {}
        self._model_skip_limits: Dict[str, int] = {}
        for provider, models in settings.provider_models.items():
            if not models:
                continue
            sequence = self._build_weighted_model_sequence(models)
            self._model_cycles[provider] = itertools.cycle(sequence)
            self._model_options[provider] = models
            self._model_skip_limits[provider] = max(1, len(models) * 4)
        self._client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    @staticmethod
    def _build_weighted_model_sequence(models: List[str]) -> List[str]:
        if len(models) <= 1:
            return models
        weighted_sequence: List[str] = []
        total = len(models)
        for index, model in enumerate(models):
            weight = 2 ** (total - index - 1)
            weighted_sequence.extend([model] * weight)
        return weighted_sequence

    def _next_model(self, provider: str, skip_model: Optional[str] = None) -> str:
        sequence = self._model_cycles[provider]
        model = next(sequence)
        if skip_model and model == skip_model:
            limit = self._model_skip_limits.get(
                provider,
                max(1, len(self._model_options.get(provider, [])) * 4),
            )
            for _ in range(limit):
                model = next(sequence)
                if model != skip_model:
                    break
        return model

    async def close(self) -> None:
        await self._client.aclose()

    # Motivation vs Logic
    # Motivation: balance traffic across provider models and recover quickly from rate limits.
    # Logic: reuse the same lock to rotate API keys and the weighted model cycle, skipping variants that just produced 429s.
    async def _lease(self, provider: str, skip_model: Optional[str] = None) -> tuple[str, AsyncRateLimiter, str]:
        async with self._lock:
            if provider not in self._key_cycles or provider not in self._model_cycles:
                raise RuntimeError(f"No API key or model configured for provider '{provider}'.")
            idx, key = next(self._key_cycles[provider])
            limiter = self._limiters[provider][idx]
            selected_model = self._next_model(provider, skip_model)
        await limiter.acquire()
        return key, limiter, selected_model

    async def generate(self, model: TargetModel, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        provider = MODEL_PROVIDER[model]
        configured_models = self._model_options.get(provider)
        if not configured_models:
            raise RuntimeError(f"No models configured for provider '{provider}'.")
        attempts = 0
        # Root Cause vs Logic:
        # Root Cause: with a single configured model alias, the old logic retried only once
        # on HTTP 429 and failed the full provider immediately.
        # Logic: retry across multiple 429 windows using env-configured retry attempts, rotating
        # model aliases when available and backing off between attempts.
        max_attempts = max(1, settings.retry_attempts) * max(1, len(configured_models))
        skip_model: Optional[str] = None
        last_exception: Optional[Exception] = None
        while attempts < max_attempts:
            key, limiter, selected_model = await self._lease(provider, skip_model=skip_model)
            try:
                if provider == "openai":
                    return await self._call_openai(key, selected_model, messages, max_tokens)
                if provider == "xai":
                    return await self._call_xai(key, selected_model, messages, max_tokens)
                if provider == "google":
                    return await self._call_google(key, selected_model, messages, max_tokens)
                if provider == "mistral":
                    return await self._call_mistral(key, selected_model, messages, max_tokens)
                if provider == "claude":
                    return await self._call_claude(key, selected_model, messages, max_tokens)
                raise RuntimeError(f"Unsupported provider: {provider}")
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                is_rate_limit = status_code == 429
                is_transient_server = 500 <= status_code < 600
                is_service_unavailable = status_code == 503
                if is_rate_limit or is_transient_server:
                    limiter.backoff()
                    attempts += 1
                    last_exception = exc
                    has_variants = len(configured_models) > 1
                    skip_model = selected_model if has_variants else None
                    if is_service_unavailable:
                        delay_seconds = settings.service_unavailable_retry_delay_seconds
                        reason = "service unavailable"
                        # Motivation vs Logic:
                        # Motivation: transient 503 responses protect a failing endpoint and need a full-window cooldown.
                        # Logic: pause for the SERVICE_UNAVAILABLE_RETRY_DELAY_SECONDS before the next retry while still rotating keys/models.
                    else:
                        delay_seconds = settings.retry_base_delay_seconds * (2 ** min(attempts - 1, 5))
                        reason = "rate limit" if is_rate_limit else "server error"
                    # Root Cause vs Logic:
                    # Root Cause: 5xx outages immediately terminated the benchmark run, leaving partial progress.
                    # Logic: treat transient server errors like rate limits so we back off, rotate models/keys,
                    # and retry until the configured cap is reached.
                    logger.warning(
                        "Model %s for %s hit %s (%d) (attempt %d/%d); retrying in %.1fs.",
                        selected_model,
                        provider,
                        reason,
                        status_code,
                        attempts,
                        max_attempts,
                        delay_seconds,
                    )
                    if attempts >= max_attempts:
                        break
                    await asyncio.sleep(delay_seconds)
                    continue
                raise
        raise RuntimeError(f"All configured models for '{provider}' hit rate limit after {max_attempts} attempts.") from last_exception

    async def _call_openai(self, key: str, model: str, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        payload = {
            "model": model,
            "input": [{"role": item["role"], "content": [{"type": "input_text", "text": item["content"]}]} for item in messages],
            "max_output_tokens": max_tokens,
        }
        response = await self._client.post(
            f"{settings.openai_base_url}/responses",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        text = norm_text(data.get("output_text"))
        if not text:
            chunks: List[str] = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        chunks.append(content.get("text", ""))
            text = norm_text(" ".join(chunks))
        return ProviderResponse(text=text, raw=data)

    async def _call_xai(self, key: str, model: str, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        response = await self._client.post(
            f"{settings.xai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        text = norm_text(data["choices"][0]["message"].get("content", ""))
        return ProviderResponse(text=text, raw=data)

    async def _call_google(self, key: str, model: str, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        system_parts = [item["content"] for item in messages if item["role"] == "system"]
        content_parts = []
        for item in messages:
            if item["role"] == "system":
                continue
            role = "model" if item["role"] == "assistant" else "user"
            content_parts.append({"role": role, "parts": [{"text": item["content"]}]})
        payload: Dict[str, Any] = {
            "contents": content_parts,
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
        }
        if system_parts:
            payload["system_instruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        response = await self._client.post(
            f"{settings.google_base_url}/models/{model}:generateContent",
            params={"key": key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        text_parts: List[str] = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
        return ProviderResponse(text=norm_text(" ".join(text_parts)), raw=data)

    async def _call_mistral(self, key: str, model: str, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        response = await self._client.post(
            f"{settings.mistral_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = str(content)
        return ProviderResponse(text=norm_text(text), raw=data)

    # Motivation vs Logic:
    # Motivation: Claude uses Anthropic's Messages API shape instead of the chat-completions payload
    # used by several other providers in this app.
    # Logic: split system prompts into Anthropic's top-level `system` field, pass the remaining
    # conversation through `messages`, and normalize the returned text blocks into the shared response type.
    async def _call_claude(self, key: str, model: str, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        system_parts = [item["content"] for item in messages if item["role"] == "system"]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in messages
                if item["role"] != "system"
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        response = await self._client.post(
            f"{settings.claude_base_url}/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        text = " ".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return ProviderResponse(text=norm_text(text), raw=data)
