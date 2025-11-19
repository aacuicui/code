import os
import asyncio
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()
# 加载.env文件
api_key = os.getenv("API_KEY")
## --- 配置 ---
## 确保设置了您的 API 密钥环境变量（例如，OPENAI_API_KEY）
if not api_key:
    raise ValueError("OPENAI_API_KEY 环境变量未设置")
else:
    # 尝试初始化Qwen模型
    try:
        # 使用OpenAI兼容接口调用Qwen模型
        llm = ChatOpenAI(
            model="qwen-plus",  # 或其他Qwen模型，如qwen-turbo, qwen-max等
            openai_api_key=api_key,  # 直接使用已获取的api_key变量
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0
        )
        print(f"Qwen语言模型已初始化: {llm.model_name}")
    except Exception as e:
        print(f"初始化Qwen语言模型时出错: {e}")
        llm = None

## --- 定义独立链 ---
## 这三个链代表可以并行执行的不同任务。
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "简洁地总结以下主题："),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "生成关于以下主题的三个有趣问题："),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "从以下主题中识别 5-10 个关键术语，用逗号分隔："),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

## --- 构建并行 + 综合链 ---
## 1. 定义要并行运行的任务块。这些的结果，
##    以及原始主题，将被馈送到下一步。
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),  # 传递原始主题
    }
)

## 2. 定义将组合并行结果的最终综合提示词。
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """基于以下信息：
    摘要：{summary}
    相关问题：{questions}
    关键术语：{key_terms}
    综合一个全面的答案。"""),
    ("user", "原始主题：{topic}")
])

## 3. 通过将并行结果直接管道化
##    到综合提示词，然后是 LLM 和输出解析器，构建完整链。
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

## --- 运行链 ---
async def run_parallel_example(topic: str) -> None:
    """
    异步调用具有特定主题的并行处理链
    并打印综合结果。
    参数：
        topic: 要由 LangChain 链处理的输入主题。
    """
    if not llm:
        print("LLM 未初始化。无法运行示例。")
        return
    
    print(f"\n--- 运行主题的并行 LangChain 示例：'{topic}' ---")
    try:
        # `ainvoke` 的输入是单个 'topic' 字符串，
        # 然后传递给 `map_chain` 中的每个可运行对象。
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- 最终响应 ---")
        print(response)
    except Exception as e:
        print(f"\n链执行期间发生错误：{e}")

if __name__ == "__main__":
    test_topic = "太空探索的历史"
    # 在 Python 3.7+ 中，asyncio.run 是运行异步函数的标准方式。
    asyncio.run(run_parallel_example(test_topic))