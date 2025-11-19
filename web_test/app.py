import os
import asyncio
from typing import Optional
from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 初始化Flask应用
app = Flask(__name__)

# 全局变量存储LLM实例
llm = None

def initialize_llm():
    """初始化语言模型"""
    global llm
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        raise ValueError("API_KEY 环境变量未设置")
    
    try:
        llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_key=api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0
        )
        print(f"Qwen语言模型已初始化: {llm.model_name}")
        return True
    except Exception as e:
        print(f"初始化Qwen语言模型时出错: {e}")
        return False

# 初始化LangChain链
def create_chains():
    """创建LangChain处理链"""
    
    # 定义一个“总结链”（summarize_chain），其类型为 Runnable（LangChain 的可运行对象）
    summarize_chain: Runnable = (
        # 1. 构造提示模板：把 system 角色和 user 角色拼成对话格式
        ChatPromptTemplate.from_messages([
            ("system", "简洁地总结以下主题："),   # 系统提示，告诉模型要“简洁总结”
            ("user", "{topic}")                # 用户占位符，运行时会被实际主题替换
        ])
        # 2. 用管道符“|”把提示模板传递给大模型 llm，生成回复
        | llm
        # 3. 再用管道符把模型输出传给 StrOutputParser()，把模型返回的消息对象解析成纯字符串
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
    
    trend_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "以积极的态度分析以下主题："),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    # 构建并行处理链
    map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "trend": trend_chain,
            "topic": RunnablePassthrough(),  # 把输入的原始主题原样透传下去，供后续链（如terms_chain中的{topic}）使用
        }
    )

    # 综合结果链
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """基于以下信息：
        摘要：{summary}
        相关问题：{questions}
        关键术语：{key_terms}
        情感倾向：{trend}
        综合一个全面的答案。"""),
        ("user", "原始主题：{topic}")
    ])

    full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()
    
    return full_parallel_chain

# 异步处理函数
async def process_topic_async(topic: str, chain):
    """异步处理主题"""
    try:
        response = await chain.ainvoke(topic)
        return {
            "success": True,
            "result": response,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }

# Flask路由
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_topic():
    """处理用户输入的主题"""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({
                "success": False,
                "error": "请输入主题"
            })
        
        if not llm:
            return jsonify({
                "success": False,
                "error": "语言模型未正确初始化"
            })
        
        # 创建处理链
        chain = create_chains()
        
        # 异步处理
        async def run_processing():
            return await process_topic_async(topic, chain)
        
        # 运行异步任务
        result = asyncio.run(run_processing())
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"处理过程中发生错误: {str(e)}"
        })

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "ok",
        "llm_initialized": llm is not None
    })

if __name__ == '__main__':
    # 初始化语言模型
    if initialize_llm():
        print("应用启动成功！")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("应用启动失败：语言模型初始化失败")