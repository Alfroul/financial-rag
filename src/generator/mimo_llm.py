import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMAuthError(LLMError):
    pass


class LLMQuotaError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class MimoLLM:
    """MiMo LLM（OpenAI 兼容接口），用于 RAG 生成和 Query Rewriting。"""

    def __init__(
        self,
        api_key: str,
        model: str = "mimo-v2-pro",
        base_url: str = "https://api.xiaomimimo.com",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": stream,
        }

    @staticmethod
    def _prepare_messages(system_prompt: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        all_msgs: list[dict[str, str]] = []
        if system_prompt:
            all_msgs.append({"role": "system", "content": system_prompt})
        all_msgs.extend(messages)
        return all_msgs

    def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        all_msgs = self._prepare_messages(system_prompt, messages)

        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=self._build_payload(all_msgs, temperature, max_tokens),
                    timeout=60.0,
                )
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"] or ""
            except Exception as e:
                logger.warning("MiMo API 调用失败 (%d/3): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"MiMo API 连续 3 次调用失败: {self._model}")

    async def achat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        all_msgs = self._prepare_messages(system_prompt, messages)

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=self._build_headers(),
                        json=self._build_payload(all_msgs, temperature, max_tokens),
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    return body["choices"][0]["message"]["content"] or ""
                except Exception as e:
                    logger.warning("MiMo API 调用失败 (%d/3): %s", attempt + 1, e)
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"MiMo API 连续 3 次调用失败: {self._model}")

    def stream_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        all_msgs = self._prepare_messages(system_prompt, messages)

        with httpx.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._build_headers(),
            json=self._build_payload(all_msgs, temperature, max_tokens, stream=True),
            timeout=60.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield delta["content"]

    async def astream_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        all_msgs = self._prepare_messages(system_prompt, messages)

        async with httpx.AsyncClient(timeout=60.0) as client, client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._build_headers(),
            json=self._build_payload(all_msgs, temperature, max_tokens, stream=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield delta["content"]
