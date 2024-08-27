import tiktoken

from graphrag.query.llm.base import BaseLLM
from graphrag.query.structured_search.direct.search import Direct
from webserver.configs import settings


async def build_direct_engine(llm: BaseLLM, token_encoder: tiktoken.Encoding | None = None) -> Direct:
    llm_params = {
        "max_tokens": settings.local_search.llm_max_tokens,
        "temperature": settings.local_search.temperature,
        "top_p": settings.local_search.top_p,
        "n": settings.local_search.n,
    }
    return Direct(llm, token_encoder, llm_params=llm_params)
