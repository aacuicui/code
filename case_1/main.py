
# 导入必要的库
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_community.chat_models import ChatTongyi
# --- Qwen模型配置选项（选择其中一种方式）---

# 方式1：通过OpenAI兼容接口（推荐，最简单）
from langchain_openai import ChatOpenAI

# 获取并验证DASHSCOPE_API_KEY
# 获取API密钥
api_key = os.getenv("API_KEY")

# 初始化模型或设置为None
if not api_key:
    llm = None
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


## --- 定义模拟子Agent处理程序（保持不变）---
def booking_handler(request: str) -> str:
    """模拟预订Agent处理请求。"""
    print("\n--- 委托给预订处理程序 ---")
    return f"预订处理程序处理了请求：'{request}'。结果：模拟预订操作。"

def info_handler(request: str) -> str:
    """模拟信息Agent处理请求。"""
    print("\n--- 委托给信息处理程序 ---")
    return f"信息处理程序处理了请求：'{request}'。结果：模拟信息检索。"

def unclear_handler(request: str) -> str:
    """处理无法委托的请求。"""
    print("\n--- 处理不清楚的请求 ---")
    return f"协调器无法委托请求：'{request}'。请澄清。"

## --- 定义协调器路由链（针对Qwen模型优化提示词）---
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个请求路由协调器。请分析用户的请求并确定哪个专家处理程序应该处理它。

请严格按照以下规则分类：
- 如果请求与预订、预约、订购、下单相关（如航班、酒店、餐厅等），输出 'booker'
- 如果请求是询问信息、查询、咨询、了解知识，输出 'info'  
- 如果请求不清楚或不属于以上任何类别，输出 'unclear'

请只输出一个单词：'booker'、'info' 或 'unclear'，不要添加任何其他内容。"""),
    ("user", "{request}")
])

if llm:
    # 使用“|”运算符把 prompt、llm 和输出解析器串成一条链（LangChain 的 LCEL 语法）
    # 1. coordinator_router_prompt 先把用户请求渲染成最终发给模型的完整提示
    # 2. llm 接收提示并调用大模型，返回 AIMessage
    # 3. StrOutputParser() 把 AIMessage 中的内容字段提取出来，变成纯字符串
    # 整体效果：输入 {"request": "..."} → 输出模型回答的字符串（如 "booker"/"info"/"unclear"）
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

## --- 定义委托逻辑（保持不变）---
# branches 字典：根据路由决策（booker/info/unclear）把请求分发给对应的“子Agent”处理函数
# 每个分支都用 RunnablePassthrough.assign 把子Agent返回的字符串挂到 output 字段，
# 方便后续统一通过 x['output'] 拿到最终结果。
# 注意：x['request']['request'] 是因为外层链把原始请求包了两层，第一层是 RunnablePassthrough 的输入键。
branches = {
    "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])),
    "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])),
    "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])),
}

delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip().lower() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip().lower() == 'info', branches["info"]),
    branches["unclear"]
)
#协调器主链
# 1. 先通过 coordinator_router_chain 路由到 booker/info/unclear
# 2. 根据路由结果，通过 delegation_branch 分发给对应的子Agent处理
# 3. 最后通过 lambda x: x['output'] 提取子Agent的处理结果
coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output'])

## --- 示例用法 ---
def main():
    if not llm:
        print("\n由于LLM初始化失败，跳过执行。")
        # 尝试使用备用方案
        print("请检查API_KEY环境变量是否设置正确")
        return
    # model=ChatTongyi(
    #     model="qwen-plus-latest"
    # )
    # print(model.invoke("你好"))

    print("--- 运行预订请求 ---")
    request_a = "给我预订去伦敦的航班。"
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"最终结果A: {result_a}")
    
    print("\n--- 运行信息请求 ---")
    request_b = "意大利的首都是什么？"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"最终结果B: {result_b}")
    
    print("\n--- 运行不清楚的请求 ---")
    request_c = "关于量子物理学的事。"
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"最终结果C: {result_c}")
    
    print("\n--- 运行不清楚的请求 ---")
    request_d = "今天吃了晚饭。"
    result_d = coordinator_agent.invoke({"request": request_d})
    print(f"最终结果D: {result_d}")

if __name__ == "__main__":
    main()