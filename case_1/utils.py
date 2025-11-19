## Copyright (c) 2025 Marco Fago
#
## 此代码根据 MIT 许可证授权。
## 请参阅仓库中的 LICENSE 文件以获取完整许可文本。
import uuid
from typing import Dict, Any, Optional
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event

## --- 定义工具函数 ---
## 这些函数模拟专家 Agent 的操作。
def booking_handler(request: str) -> str:
    """
    处理航班和酒店的预订请求。
    参数：
        request: 用户的预订请求。
    返回：
        确认预订已处理的消息。
    """
    print("-------------------------- 调用预订处理程序 ----------------------------")
    return f"已模拟对 '{request}' 的预订操作。"

def info_handler(request: str) -> str:
    """
    处理一般信息请求。
    参数：
        request: 用户的问题。
    返回：
        表示信息请求已处理的消息。
    """
    print("-------------------------- 调用信息处理程序 ----------------------------")
    return f"对 '{request}' 的信息请求。结果：模拟信息检索。"

def unclear_handler(request: str) -> str:
    """处理无法委托的请求。"""
    return f"协调器无法委托请求：'{request}'。请澄清。"

## --- 从函数创建工具 ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

## 定义配备各自工具的专门子 Agent
booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="一个专门的 Agent，通过调用预订工具处理所有航班和酒店预订请求。",
    tools=[booking_tool]
)

info_agent = Agent(
    name="Info",
    model="gemini-2.0-flash",
    description="一个专门的 Agent，通过调用信息工具提供一般信息并回答用户问题。",
    tools=[info_tool]
)

## 定义具有明确委托指令的父 Agent
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "你是主协调器。你唯一的任务是分析传入的用户请求"
        "并将它们委托给适当的专家 Agent。不要尝试直接回答用户。\n"
        "- 对于任何与预订航班或酒店相关的请求，委托给 'Booker' Agent。\n"
        "- 对于所有其他一般信息问题，委托给 'Info' Agent。"
    ),
    description="一个将用户请求路由到正确专家 Agent 的协调器。",
    # sub_agents 的存在默认启用 LLM 驱动的委托（自动流）。
    sub_agents=[booking_agent, info_agent]
)

## --- 执行逻辑 ---
async def run_coordinator(runner: InMemoryRunner, request: str):
    """使用给定请求运行协调器 Agent 并委托。"""
    print(f"\n--- 使用请求运行协调器: '{request}' ---")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            if event.is_final_response() and event.content:
                # 尝试直接从 event.content 获取文本
                # 以避免迭代部分
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    # 后备：迭代部分并提取文本（可能触发警告）
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                # 假设循环应在最终响应后中断
                break
        
        print(f"协调器最终响应: {final_result}")
        return final_result
    except Exception as e:
        print(f"处理您的请求时出错: {e}")
        return f"处理您的请求时出错: {e}"

async def main():
    """运行 ADK 示例的主函数。"""
    print("--- Google ADK 路由示例（ADK 自动流风格）---")
    print("注意：这需要安装并认证 Google ADK。")
    
    runner = InMemoryRunner(coordinator)
    
    # 示例用法
    result_a = await run_coordinator(runner, "给我在巴黎预订一家酒店。")
    print(f"最终输出 A: {result_a}")
    
    result_b = await run_coordinator(runner, "世界上最高的山是什么？")
    print(f"最终输出 B: {result_b}")
    
    result_c = await run_coordinator(runner, "告诉我一个随机事实。") # 应该去 Info
    print(f"最终输出 C: {result_c}")
    
    result_d = await run_coordinator(runner, "查找下个月去东京的航班。") # 应该去 Booker
    print(f"最终输出 D: {result_d}")

if __name__ == "__main__":
    import nest_asyncio
    import asyncio
    nest_asyncio.apply()
    asyncio.run(main())