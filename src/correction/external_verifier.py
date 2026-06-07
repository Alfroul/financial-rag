from __future__ import annotations

import json
import logging

import httpx

from src.correction.types import ClaimVerdict

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是金融文档事实核查审计员。逐条判断以下断言是否被证据文本直接支撑。"
    "严格标准：仅当证据包含直接支持断言的具体信息时判定为 YES。\n\n"
    "逐条输出 JSON 数组，每项含 claim, supported(true/false), "
    "evidence_quote(引用原文或\"UNSUPPORTED\")。"
    "只输出 JSON 数组，不要输出其他内容。"
)


class ExternalVerifier:
    """Layer 4: External model (mimo-v2.5-pro) verification."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        model: str = "mimo-v2.5-pro",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def verify(self, claims: list[str], context: str) -> list[ClaimVerdict]:
        """Verify claims using external model via OpenAI-compatible API.

        On API failure, degrades gracefully by returning all claims as SUPPORTED
        so the pipeline is never blocked by external service issues.
        """
        if not claims:
            return []

        claims_formatted = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        user_prompt = (
            f"证据：\n{context}\n\n"
            f"待判定断言：\n{claims_formatted}"
        )

        try:
            return self._call_api(claims, user_prompt)
        except Exception as e:
            logger.warning("ExternalVerifier API call failed, degrading to all SUPPORTED: %s", e)
            return [
                ClaimVerdict(
                    claim=c,
                    supported=True,
                    evidence="UNSUPPORTED (API failure, degraded)",
                    source="external_verifier_degraded",
                )
                for c in claims
            ]

    def _call_api(self, claims: list[str], user_prompt: str) -> list[ClaimVerdict]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 1024,
        }

        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()

        body = response.json()
        content = body["choices"][0]["message"]["content"]

        return self._parse_response(claims, content)

    def _parse_response(self, claims: list[str], content: str) -> list[ClaimVerdict]:
        json_str = content.strip()
        # LLMs may wrap JSON in markdown code fences
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("ExternalVerifier: failed to parse model response as JSON, degrading")
            return [
                ClaimVerdict(
                    claim=c,
                    supported=True,
                    evidence="UNSUPPORTED (parse failure, degraded)",
                    source="external_verifier_degraded",
                )
                for c in claims
            ]

        if not isinstance(parsed, list):
            logger.warning("ExternalVerifier: unexpected response format, degrading")
            return [
                ClaimVerdict(claim=c, supported=True, evidence="", source="external_verifier")
                for c in claims
            ]

        results: list[ClaimVerdict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            claim_text = str(item.get("claim", ""))
            supported = bool(item.get("supported", True))
            evidence = str(item.get("evidence_quote", ""))
            results.append(ClaimVerdict(
                claim=claim_text,
                supported=supported,
                evidence=evidence,
                source="external_verifier",
            ))

        if not results:
            results = [
                ClaimVerdict(claim=c, supported=True, evidence="", source="external_verifier")
                for c in claims
            ]

        logger.info(
            "ExternalVerifier: %d/%d claims supported",
            sum(1 for r in results if r.supported),
            len(results),
        )
        return results
