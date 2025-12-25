import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
import socket

def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


class PredictionRequest(BaseModel):
    prompt: str

class PredictionResponse(BaseModel):
    response: str

# 设置环境变量，启用 vLLM 的 ModelScope 支持
os.environ['VLLM_USE_MODELSCOPE']='True'

# 全局计数器
count = 1

# 模型本地路径映射字典
model_local_dict = {
    "Qwen/Qwen3-1.7B": "/app/Qwen/Qwen3-1.7B"
}

@asynccontextmanager
async def initializationEngine(app: FastAPI):
    '''
        初始化引擎
    '''
    print("Initializing vLLM engine...")
    try:
        # 定义system prompt
        #system_prompt = """你是一名专业的GPU编程教材内容整理助手，专门从《Programming Massively Parallel Processors》教材及相关GPU架构资料中提取和整理知识点。
        #                请严格遵守以下内容生成规则：
        #                一、问题设计规范：
        #                1. 每个问题必须独立、完整、自包含
        #                2. 禁止任何形式的上下文依赖
        #                3. 问题必须清晰具体
        #                二、回答内容规范：
        #                1. 技术准确性：严格基于教材公认知识
        #                2. 解释清晰性：语言简洁明了，面向学生
        #                3. 信息完整性：回答必须自成体系
        #                三、主题范围（专注以下核心领域）：
        #                1. GPU架构基础
        #                2. CUDA编程模型  
        #                3. 并行算法与模式
        #                4. 性能优化技术
        #                5. 内存层次结构
        #                请保持风格：技术准确 + 简明易懂 + 教学导向。"""
        system_prompt = """你是一名专业的GPU编程教材内容整理助手，请直接回答问题，**只回答一次**，不要续写问题本身，不要添加任何额外的问题或评估。"""
        engine_args = AsyncEngineArgs(
            model=model_local_dict["Qwen/Qwen3-1.7B"],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            trust_remote_code=True
        )
        app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
        app.state.system_prompt = system_prompt
        print("vLLM engine initialized successfully!")
    except Exception as e:
        print(f"Engine initialization failed: {e}")
        raise
    yield
    print("Shutting down vLLM engine...")

app = FastAPI(title="vLLM Service", lifespan=initializationEngine)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    文本生成预测接口
    Example:
        >>> 请求
        {"prompt": "今天天气很好，"}
        
        >>> 响应  
        {"response": "适合出去散步。"}
        
    Notes:
        - 使用全局计数器(count)确保每次生成使用不同的请求ID
        - 生成的文本会去除首尾空白字符
    """
    global count
    engine = app.state.engine
    prompt_text = request.prompt
    # raise RuntimeError(request.prompt)
    
    system_prompt = app.state.system_prompt
    # 完整的prompt
    formatted_prompt = f"""{system_prompt}
请根据以上角色要求，回答以下问题：
问题：{prompt_text}
回答："""

    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=50,
        max_tokens=300,
        stop=["\n问题：",  # 停止新问题
            "\n问：",  # 停止新问题
            "\n你是否",  # 停止"你是否..."
            "\n好的，",  # 停止"好的，现在请你..."
            "\n接下来",  # 停止"接下来..."
            "？\n",  # 问号后换行就停止
            "?\n",  # 英文问号后换行停止
            # 中文自问自答模式
            "\n这个描述",
            "\n上述描述",
            "\n这个回答",
            "\n以上回答",
            "\n是否正确",
            "\n是否准确",
            "\n是否错误",
            "\n如果有误",
            "\n如果错误",
            "\n请指出",
            "\n\n现在",
            "\n\n这",
            "###",  # 添加更多停止词
            "---",
            "'''",
            "<|endoftext|>",],
        repetition_penalty=1.3,  # 增加重复惩罚
        frequency_penalty=0.5,   # 添加频率惩罚
    )
    results_generator = engine.generate(formatted_prompt, sampling_params, str(count))
    count += 1
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    generated_text = final_output.outputs[0].text

    # --- 网络连通性测试 ---
    internet_ok = check_internet()
    print("【Internet Connectivity Test】:",
        "CONNECTED" if internet_ok else "OFFLINE / BLOCKED")

    return PredictionResponse(response=generated_text.strip())

@app.get("/")
def health_check():
    '''
        在health_check阶段会执行 `initializationEngine` 函数，
        Timeout: 180s
    '''
    return {"status": "ok"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)