from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# --- 关键路径配置 ---
# 1. 基础模型路径
base_model_path = "/root/notebook/qinglan/model/Qwen/Qwen3-1.7B"
# 2. LoRA补丁路径
lora_path = "/root/notebook/qinglan/train_model/model_outputs/checkpoint-400"
# 3. 融合后的模型存放位置
output_path = "/root/notebook/qinglan/train_model/modelscope/400"
# ------------------

print(f"正在加载基础模型: {base_model_path} ...")
# 使用 CPU 加载以防止显存不够
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu", 
    trust_remote_code=True
)

print(f"正在加载 LoRA 权重: {lora_path} ...")

# 加载微调补丁
model = PeftModel.from_pretrained(base_model, lora_path)

print("正在执行模型融合 (Merge and Unload)...")
model = model.merge_and_unload()

print(f"正在保存新模型到: {output_path} ...")
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path, safe_serialization=True)  # 添加safe_serialization

print("正在保存 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("🎉 模型融合完成！")
print(f"模型已保存到: {output_path}")