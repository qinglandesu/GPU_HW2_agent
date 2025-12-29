import os
import uvicorn
import socket
import asyncio
from typing import Union, List
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

class PredictionRequest(BaseModel):
    prompt: Union[str, List[str]]

class PredictionResponse(BaseModel):
    response: Union[str, List[str]]

# 设置环境变量，启用 vLLM 的 ModelScope 支持
os.environ['VLLM_USE_MODELSCOPE'] = 'True'

# 全局计数器
count = 1

# 模型本地路径映射字典
model_local_dict = {
    "Qwen/Qwen3-1.7B": "/app/Qwen/Qwen3-1.7B",
    "GPUclass_qwen0": "/app/qinglandesu/GPUclass_qwen0",
}

@asynccontextmanager
async def initializationEngine(app: FastAPI):
    '''
    初始化引擎
    '''
    print("Initializing vLLM engine...")
    try:
        system_prompt = """你是一名专业的GPU编程教材内容整理助手，请直接回答问题，**只回答一次**，不要续写问题本身，不要添加任何额外的问题或评估。"""
        engine_args = AsyncEngineArgs(
            model=model_local_dict["GPUclass_qwen0"],
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_num_seqs=256,  # 增加最大并发序列数以支持batch
            max_model_len=1024  # 设置最大模型长度
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

async def generate_single(engine, formatted_prompt: str, sampling_params: SamplingParams, request_id: str) -> str:
    """
    单个prompt的生成函数
    """
    results_generator = engine.generate(formatted_prompt, sampling_params, request_id)
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    if final_output and final_output.outputs:
        return final_output.outputs[0].text.strip()
    return ""

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    文本生成预测接口（支持batch）
    
    Example - 单个输入:
        >>> 请求
        {"prompt": "什么是CUDA？"}
        
        >>> 响应  
        {"response": "CUDA是NVIDIA推出的并行计算平台和编程模型。"}
        
    Example - 批量输入:
        >>> 请求
        {"prompt": ["什么是CUDA？", "GPU和CPU有什么区别？"]}
        
        >>> 响应  
        {"response": ["CUDA是NVIDIA推出的并行计算平台和编程模型。", 
                     "GPU是专门为图形处理和并行计算设计的处理器..."]}
        
    Notes:
        - 使用全局计数器(count)确保每次生成使用不同的请求ID
        - 生成的文本会去除首尾空白字符
        - 支持单个prompt和批量prompt处理
    """
    global count
    engine = app.state.engine
    
    # 判断是单个prompt还是批量prompt
    is_batch = isinstance(request.prompt, list)
    prompts = [request.prompt] if not is_batch else request.prompt
    
    system_prompt = app.state.system_prompt
    
    # 格式化所有prompts
    formatted_prompts = []
    for prompt_text in prompts: #
        formatted_prompt = f"""{system_prompt}

现在请回答以下问题：

###问题:
{prompt_text}

###回答:"""
        formatted_prompts.append(formatted_prompt)
    
    # 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.2,
        #top_p=0.9,
        #top_k=40,
        max_tokens=300,
        stop=[
            "\n问题：", "\n问：", "\n你是否", "\n好的，", "\n接下来",
            "\n这个描述", "\n上述描述", "\n这个回答", "\n以上回答", "\n请指出",
            "\n\n现在", "\n\n这个", "\n\n好的", "\n\n注意",
            "仅回答", "（回答", "(回答", "（字数", "(字数", "（注", "(注",
            "###", "---", "'''", "```", "<|endoftext|>",
        ],
        #repetition_penalty=1.3,
        #frequency_penalty=0.5,
    )
    
    # 生成请求ID列表
    request_ids = [str(count + i) for i in range(len(formatted_prompts))]
    count += len(formatted_prompts)
    
    # 创建生成任务列表
    generation_tasks = []
    for prompt_text, request_id in zip(formatted_prompts, request_ids):
        task = engine.generate(prompt_text, sampling_params, request_id)
        generation_tasks.append(task)
    
    # 创建并行任务 - 真正的并行处理
    tasks = []
    for formatted_prompt, request_id in zip(formatted_prompts, request_ids):
        # 为每个prompt创建异步任务
        task = asyncio.create_task(
            generate_single(engine, formatted_prompt, sampling_params, request_id)
        )
        tasks.append(task)
    
    # 等待所有任务完成（并行执行）
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常结果
    processed_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Error generating response for prompt {i}: {response}")
            processed_responses.append(f"Error: {str(response)}")
        else:
            processed_responses.append(response)
    
    # --- 网络连通性测试 ---
    internet_ok = check_internet()
    print("【Internet Connectivity Test】:",
          "CONNECTED" if internet_ok else "OFFLINE / BLOCKED")
    print(f"【Request Processed】: Batch size = {len(prompts)}, Parallel tasks = {len(tasks)}, Is batch = {is_batch}")
    
    # 如果是单个请求，返回单个字符串
    if not is_batch:
        return PredictionResponse(response=processed_responses[0])
    
    # 如果是批量请求，返回字符串列表
    return PredictionResponse(response=processed_responses)


@app.get("/")
def health_check():
    '''
    health_check接口
    '''
    return {"status": "batch"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)