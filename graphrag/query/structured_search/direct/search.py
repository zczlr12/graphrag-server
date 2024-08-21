# 自定义
# Licensed under the MIT License

"""Direct implementation."""

from typing import Any

import tiktoken

from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.structured_search.base import BaseSearch, SearchResult

DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}


class Direct(BaseSearch):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        llm: BaseLLM,
        token_encoder: tiktoken.Encoding | None = None,
        callbacks: list[BaseLLMCallback] | None = None,
        llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS
    ):
        super().__init__(
            llm=llm,
            token_encoder=token_encoder,
            llm_params=llm_params,
        )
        self.callbacks = callbacks

    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory,
        **kwargs
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        try:
            search_messages = [
                {
                    "role": turn.role,
                    "content": turn.content
                } for turn in conversation_history.turns
            ]
            search_messages.append({"role": "user", "content": query})

            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data="",
                context_text="",
                completion_time=0,
                llm_calls=1,
                prompt_tokens=0
            )

        except Exception:
            return SearchResult(
                response="",
                context_data="",
                context_text="",
                completion_time=0,
                llm_calls=1,
                prompt_tokens=0
            )

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory,
        **kwargs
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user question."""
        try:
            search_messages = [
                {
                    "role": turn.role,
                    "content": turn.content
                } for turn in conversation_history.turns
            ]
            search_messages.append({"role": "user", "content": query})

            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data="",
                context_text="",
                completion_time=0,
                llm_calls=1,
                prompt_tokens=0
            )

        except Exception:
            return SearchResult(
                response="",
                context_data="",
                context_text="",
                completion_time=0,
                llm_calls=1,
                prompt_tokens=0
            )
