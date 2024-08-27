from typing import Any, Literal

from pydantic import BaseModel


class ChatCompletionMessageParam(BaseModel):
    content: str
    role: str = "user"


class ResponseFormat(BaseModel):
    type: str


class ChatCompletionStreamOptionsParam(BaseModel):
    enable: bool


class ChatCompletionToolParam(BaseModel):
    name: str
    description: str


class CompletionCreateParamsBase(BaseModel):
    messages: list[ChatCompletionMessageParam]
    model: str
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    max_tokens: int | None = None
    n: int | None = None
    parallel_tool_calls: bool = False
    presence_penalty: float | None = None
    response_format: ResponseFormat = ResponseFormat(type="text")
    seed: int | None = None
    service_tier: Literal["auto", "default"] | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    temperature: float | None = 0.0
    tools: list[ChatCompletionToolParam] = None
    top_logprobs: int | None = None
    top_p: float | None = 1.0
    user: str | None = None
    community_level: int | None = 2
    response_type: str | None = None

    def llm_chat_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "seed": self.seed
        }


class ChatQuestionGen(BaseModel):
    messages: list[ChatCompletionMessageParam]
    model: str
    max_tokens: int | None = None
    temperature: float | None = 0.0
    n: int | None = None
    community_level: int | None = 2


class Model(BaseModel):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]
