from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generator.mimo_llm import MimoLLM

logger = logging.getLogger(__name__)


def _trim_history_for_rewrite(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(chat_history) <= 6:
        return list(chat_history)
    return list(chat_history[-6:])


_QUERY_REWRITE_SYSTEM_PROMPT = """\
你是一个专业的金融领域查询改写助手。你的任务是：根据对话历史，将用户最新提出的问题改写为一个**独立、完整**的检索用问题。

## 规则
1. 保留金融专业术语（如"沪深300"、"市盈率"、"ETF"等），不要替换或简化
2. 仅做指代消解和上下文补全，不要添加用户未提及的信息
3. 如果用户的问题本身已经独立、完整，直接输出原问题，不要做任何修改
4. 只输出改写后的问题，不要输出任何解释、分析或额外文字

## 示例
对话历史：
  用户: 沪深300的成分股有哪些？
  助手: 沪深300指数成分股覆盖了沪深两市规模大、流动性好的300只股票...

用户最新问题: 那它的市盈率呢？
改写结果: 沪深300指数的市盈率是多少？

对话历史：（空）
用户最新问题: 什么是量化交易？
改写结果: 什么是量化交易？"""


class QueryRewriter:
    """用 LLM 将多轮对话中的指代性问题改写为独立检索 query。"""

    def __init__(self, llm: MimoLLM) -> None:
        self._llm = llm

    def rewrite(self, query: str, chat_history: list[dict[str, str]]) -> str:
        """将 query + history 改写为独立的检索 query。

        如果历史为空，直接返回原问题。
        改写失败时降级为原始 query，不影响正常流程。

        Args:
            query: 用户最新提出的问题。
            chat_history: 历史对话消息列表，每项为 {"role": ..., "content": ...}。

        Returns:
            改写后的独立问题，或原始问题（当历史为空/改写失败时）。
        """
        if not chat_history or not query.strip():
            return query

        try:
            # 构造对话上下文（仅保留最近几轮，控制 token 消耗）
            recent_history = _trim_history_for_rewrite(chat_history)
            history_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in recent_history
            )

            user_prompt = f"对话历史：\n{history_text}\n\n用户最新问题: {query}\n改写结果:"

            rewritten = self._llm.chat(
                system_prompt=_QUERY_REWRITE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            rewritten = rewritten.strip()
            # Strip common LLM output prefixes
            for prefix in ("改写结果:", "改写结果：", "改写:", "改写："):
                if rewritten.startswith(prefix):
                    rewritten = rewritten[len(prefix) :].strip()
                    break
            if not rewritten:
                logger.debug("Query rewrite 返回空结果，使用原始 query")
                return query

            logger.info("Query rewrite: '%s' → '%s'", query, rewritten)
            return rewritten

        except Exception as e:
            logger.warning("Query rewrite 失败，降级为原始 query: %s", e)
            return query

    async def arewrite(self, query: str, chat_history: list[dict[str, str]]) -> str:
        if not chat_history or not query.strip():
            return query

        try:
            recent_history = _trim_history_for_rewrite(chat_history)
            history_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in recent_history
            )

            user_prompt = f"对话历史：\n{history_text}\n\n用户最新问题: {query}\n改写结果:"

            rewritten = await self._llm.achat(
                system_prompt=_QUERY_REWRITE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            rewritten = rewritten.strip()
            for prefix in ("改写结果:", "改写结果：", "改写:", "改写："):
                if rewritten.startswith(prefix):
                    rewritten = rewritten[len(prefix) :].strip()
                    break
            if not rewritten:
                logger.debug("异步 Query rewrite 返回空结果，使用原始 query")
                return query

            logger.info("异步 Query rewrite: '%s' → '%s'", query, rewritten)
            return rewritten

        except Exception as e:
            logger.warning("异步 Query rewrite 失败，降级为原始 query: %s", e)
            return query
