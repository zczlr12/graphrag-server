"""问题分类器."""
import os
from typing import Literal

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI


ROUTER_PROMPT = """
您是将用户问题路由到local、global或direct三种选项的专家。

local和global搜索的对象是日本福岛核事故经验反馈报告。
该报告是日本政府的最终报告，由IAEA主导，分为五个技术卷，详细介绍了事故的背景、技术评估、应急响应和事故后的管理措施。各卷的主要内容为：
1.事故描述和背景：包括对事故的详细描述和背景信息。
2.安全评估：对事故的技术和安全方面进行了详细评估。
3.应急准备和响应：讨论了事故期间的应急措施和响应。
4.事故后的恢复和管理：涉及环境恢复和核材料管理。
5.总结和建议

Local 搜索方法适用于需要了解文档中提到的特定实体的问题（例如，某一特定技术的细节是什么？）。
Global 搜索方法是一种资源密集型方法，但通常可以很好地回答需要了解整个数据集的问题（例如，事故的总体影响是什么？）。
如果以上两种方法都不适用，请选择direct。
"""


with open("settings.yaml") as file:
    config = yaml.safe_load(file)["llm"]

model = AzureChatOpenAI(
    api_key=os.environ["GRAPHRAG_API_KEY"],
    azure_endpoint=config["api_base"],
    azure_deployment=config["deployment_name"],
    api_version=config["api_version"]
)


# Data model
class RouteQuery(BaseModel):
    """将用户查询路由到最适合的搜索选项。"""

    option: Literal["local", "global", "direct"] = Field(
        ...,
        description="给定用户问题，选择将其路由到local，global，或direct.",
    )


structured_llm_router = model.with_structured_output(RouteQuery)
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_PROMPT),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


def auto_select(query: str) -> str:
    """将用户问题路由到最适合的选项."""
    return question_router.invoke({"question": query}).option
