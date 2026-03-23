from __future__ import annotations

import asyncio
import itertools
from typing import Any, Dict, List, Sequence

import httpx

from app.config import settings
from app.metrics import norm_text
from app.models import MODEL_PROVIDER, ProviderResponse, TargetModel
from app.rate_limit import AsyncRateLimiter


class ProviderPool:
    def __init__(self) -> None:
        self._key_cycles: Dict[str, Any] = {}
        self._limiters: Dict[str, List[AsyncRateLimiter]] = {}
        self._lock = asyncio.Lock()
        for provider, keys in settings.provider_keys.items():
            if keys:
                self._key_cycles[provider] = itertools.cycle(enumerate(keys))
                rpm = getattr(settings, f"{provider}_requests_per_minute")
                self._limiters[provider] = [AsyncRateLimiter(rpm) for _ in keys]
        self._client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def _lease(self, provider: str) -> tuple[str, AsyncRateLimiter]:
        async with self._lock:
            if provider not in self._key_cycles:
                raise RuntimeError(f"No API key configured for provider '{provider}'.")
            idx, key = next(self._key_cycles[provider])
            limiter = self._limiters[provider][idx]
        await limiter.acquire()
        return key, limiter

    async def generate(self, model: TargetModel, messages: Sequence[Dict[str, str]], max_tokens: int) -> ProviderResponse:
        provider = MODEL_PROVIDER[model]
        key, _ = await self._lease(provider)
        if provider == "openai":
            return await self._call_openai(key, str(model), messages, max_tokens)
        if provider == "xai":
            return await self._call_xai(key, str(model), messages, max_tokens)
        if provider == "google":
            return await self._call_google(key, str(model), messages, max_tokens)
        if provider == "mistral":
            return await self._call_mistral(key, str(model), messages, max_tokens)
        raise RuntimeError(f"Unsupported provider: {provider}")

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
