import asyncio
import logging
import os
import time
import re
from typing import List, Tuple
import uuid
from typing import Generator

import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Template
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.llm.oai import ChatOpenAI, OpenAIEmbedding
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.base import SearchResult
from graphrag.query.structured_search.global_search.callbacks import GlobalSearchLLMCallback
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.direct.search import Direct
from webserver import gtypes
from webserver import search
from webserver import utils
from webserver.configs import settings
from webserver.utils import consts

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    api_key=settings.llm.api_key,
    api_base=settings.llm.api_base,
    model=settings.llm.model,
    api_type=settings.get_api_type(),
    max_retries=settings.llm.max_retries,
    azure_ad_token_provider=settings.azure_ad_token_provider(),
    deployment_name=settings.llm.deployment_name,
    api_version=settings.llm.api_version,
    organization=settings.llm.organization,
    request_timeout=settings.llm.request_timeout,
)

text_embedder = OpenAIEmbedding(
    api_key=settings.embeddings.llm.api_key,
    api_base=settings.embeddings.llm.api_base,
    api_type=settings.get_api_type(),
    api_version=settings.embeddings.llm.api_version,
    model=settings.embeddings.llm.model,
    max_retries=settings.embeddings.llm.max_retries,
    max_tokens=settings.embeddings.llm.max_tokens,
    azure_ad_token_provider=settings.azure_ad_token_provider(),
    deployment_name=settings.embeddings.llm.deployment_name,
    organization=settings.embeddings.llm.organization,
    encoding_name=settings.encoding_model,
    request_timeout=settings.embeddings.llm.request_timeout,
)

token_encoder = tiktoken.get_encoding("cl100k_base")

local_search: LocalSearch
global_search: GlobalSearch
direct: Direct
question_gen: LocalQuestionGen


class CustomSearchCallback(GlobalSearchLLMCallback):
    stop_sign = "<<STOP>><<停止>>🅿️<<STOP>><<停止>>"

    def __init__(self):
        super().__init__()
        self.token_queue = asyncio.Queue()
        self.usage: CompletionUsage | None = None

    def on_map_response_start(self, map_response_contexts: list[str]):
        super().on_map_response_start(map_response_contexts)
        return map_response_contexts

    def on_map_response_end(self, map_response_outputs: list[SearchResult]):
        super().on_map_response_end(map_response_outputs)
        return map_response_outputs

    def on_llm_new_token(self, token: str):
        """Synchronous method to handle new tokens."""
        super().on_llm_new_token(token)
        asyncio.create_task(self.async_on_llm_new_token(token))  # Schedule async method

    def on_llm_stop(self, usage: CompletionUsage):
        super().on_llm_stop(usage)
        self.usage = usage
        asyncio.create_task(self.async_on_llm_new_token(CustomSearchCallback.stop_sign))  # Schedule async method

    async def async_on_llm_new_token(self, token: str):
        """Asynchronous version to handle new tokens."""
        await self.token_queue.put(token)

    async def generate_tokens(self) -> Generator[str, None, None]:
        while True:
            token = await self.token_queue.get()
            yield token
            if token == CustomSearchCallback.stop_sign:
                break


@app.on_event("startup")
async def startup_event():
    global local_search
    global global_search
    global direct
    global question_gen
    local_search = await search.build_local_search_engine(llm, token_encoder=token_encoder)
    global_search = await search.build_global_search_engine(llm, token_encoder=token_encoder)
    direct = await search.build_direct_engine(llm, token_encoder=token_encoder)
    question_gen = await search.build_local_question_gen(llm, token_encoder=token_encoder)


async def generate_ref_links(response: str, model: str) -> str:
    it = re.findall(r"(\[Data:(\w+)\((.+?)\)\])", response)
    if it:
        base_url = f"{settings.website_address}/v1/references"
        lines = [response, "### 参考来源"]
        index_id = model.removesuffix("-global").removesuffix("-local").removesuffix("-direct")
        input_dir = os.path.join(settings.data, index_id, "artifacts")
        reference = {}
        i = 1
        for source, key, ids in it:
            reference[source] = []
            id_list = ids.replace(" ", "").split(",")
            for id in id_list:
                try:
                    if id.isdigit():
                        url = f"{base_url}/{index_id}/{key.lower()}/{id}"
                        data = await search.get_index_data(input_dir, key.lower(), int(id))
                        reference[source].append(f"<sup>[{i}]({url})</sup>")
                        lines.append(f"{i}. [{consts.KEYS[key]}{id}：{data["title"]}]({url})")
                        i += 1
                except Exception as e:
                    print(f"错误: {e}\n{consts.KEYS[key]}：{id}")
        for key, value in reference.items():
            lines[0] = lines[0].replace(key, "".join(value))
        lines[0] = lines[0].replace("</sup><sup>", "，")
        return "\n".join(lines)
    return response


async def generate_chunks(callback, request_model, future: gtypes.TypedFuture[SearchResult]):
    usage = None
    while not future.done():
        async for token in callback.generate_tokens():
            if token == CustomSearchCallback.stop_sign:
                usage = callback.usage
                break
            chunk = ChatCompletionChunk(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request_model,
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=len(callback.response) - 1,
                        finish_reason=None,
                        delta=ChoiceDelta(
                            role="assistant",
                            content=token
                        )
                    )
                ]
            )
            yield chunk.json()

    result: SearchResult = future.result()
    chunk = ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request_model,
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=len(callback.response) - 1,
                finish_reason="stop",
                delta=ChoiceDelta(
                    role="assistant",
                    # content=result.context_data["entities"].head().to_string()
                    content=await generate_ref_links(result.response, request_model)
                )
            ),
        ],
        usage=usage
    )
    yield chunk.json()


async def initialize_search(
    request: gtypes.ChatCompletionRequest,
    search: LocalSearch | GlobalSearch,
    context: str = None,
    community_level: int = 2
) -> LocalSearch | GlobalSearch:
    search.context_builder = await switch_context(context, community_level)
    if request.response_type:
        search.response_type = request.response_type
    search.llm_params.update(request.llm_chat_params())
    return search


async def handle_sync_response(request, search, conversation_history):
    result = await search.asearch(request.messages[-1].content, conversation_history=conversation_history)
    from openai.types.chat.chat_completion import Choice
    completion = ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=await generate_ref_links(result.response, request.model)
                )
            )
        ],
        usage=CompletionUsage(
            completion_tokens=-1,
            prompt_tokens=result.prompt_tokens,
            total_tokens=-1
        )
    )
    return JSONResponse(content=jsonable_encoder(completion))


async def handle_stream_response(request, search, conversation_history):
    callback = CustomSearchCallback()
    future = gtypes.TypedFuture[SearchResult]()

    async def run_search():
        result = await search.asearch(request.messages[-1].content, conversation_history)
        future.set_result(result)

    search.callbacks = [callback]
    asyncio.create_task(run_search())
    return StreamingResponse(generate_chunks(callback, request.model, future), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: gtypes.ChatCompletionRequest):
    if not local_search or not global_search or not direct:
        logger.error("graphrag search engines is not initialized")
        raise HTTPException(status_code=500, detail="graphrag search engines is not initialized")

    request.model = get_latest_model(request.model)

    try:
        conversation_history = None
        if len(request.messages) > 1:
            history = request.messages[:-1]
            conversation_history = ConversationHistory.from_list([message.dict() for message in history])

        if request.model.endswith("local"):
            search = await initialize_search(request, local_search, request.model, request.community_level)
        elif request.model.endswith("global"):
            search = await initialize_search(request, global_search, request.model, request.community_level)
        else:
            direct.llm_params.update(request.llm_chat_params())
            search = direct

        if not request.stream:
            return await handle_sync_response(request, search, conversation_history)
        else:
            return await handle_stream_response(request, search, conversation_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/advice_questions", response_model=gtypes.QuestionGenResult)
async def get_advice_question(request: gtypes.ChatQuestionGen):
    request.model = get_latest_model(request.model)

    if request.model.endswith("local"):
        local_context = await switch_context(request.model, request.community_level)
        question_gen.context_builder = local_context
    else:
        raise NotImplementedError(f"model {request.model} is not supported")
    question_history = [message.content for message in request.messages if message.role == "user"]
    candidate_questions = await question_gen.agenerate(
        question_history=question_history, context_data=None, question_count=5
    )
    # the original generated question is "- what about xxx?"
    questions: list[str] = [question.removeprefix("-").strip() for question in candidate_questions.response]
    resp = gtypes.QuestionGenResult(questions=questions,
                                    completion_time=candidate_questions.completion_time,
                                    llm_calls=candidate_questions.llm_calls,
                                    prompt_tokens=candidate_questions.prompt_tokens)
    return resp


@app.get("/v1/references/{index_id}/{datatype}/{idx}", response_class=HTMLResponse)
async def get_reference(index_id: str, datatype: str, idx: int):
    input_dir = os.path.join(settings.data, index_id, "artifacts")
    if not os.path.exists(input_dir):
        raise HTTPException(status_code=404, detail=f"{index_id} not found")
    if datatype not in ["documents", "entities", "claims", "sources", "reports", "relationships"]:
        raise HTTPException(status_code=404, detail=f"{datatype} not found")

    data = await search.get_index_data(input_dir, datatype, idx)
    html_file_path = os.path.join("webserver", "templates", f"{datatype}_template.html")
    with open(html_file_path, encoding="utf-8") as file:
        html_content = file.read()
    template = Template(html_content)
    html_content = template.render(index_id=index_id, idx=idx, data=data)
    return HTMLResponse(content=html_content)


async def switch_context(model: str, community_level: int = 2):
    if model.endswith("global"):
        input_dir = os.path.join(settings.data, model.removesuffix("-global"), "artifacts")
        context_builder = await search.load_global_context(input_dir, token_encoder, community_level)
    elif model.endswith("local"):
        input_dir = os.path.join(settings.data, model.removesuffix("-local"), "artifacts")
        context_builder = await search.load_local_context(input_dir, text_embedder, token_encoder, community_level)
    else:
        raise NotImplementedError(f"model {model} is not supported")
    return context_builder


def get_latest_model(model: str):
    if model in [consts.LATEST_MODEL_LOCAL, consts.LATEST_MODEL_GLOBAL]:
        latest_dir = utils.get_latest_subdir(settings.data)
        model = model.replace("GraphRAG-latest", latest_dir)
    return model


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=settings.server_port)
