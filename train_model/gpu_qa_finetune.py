import os
import argparse
import json
import torch
import unsloth
import transformers
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig  # 导入量化配置类，修复load_in_4bit弃用警告
)
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraConfig
from evaluate import load
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# ========== 第一步：添加精准日志过滤器（核心，彻底屏蔽目标警告） ==========
import logging
class QwenCacheWarningFilter(logging.Filter):
    def filter(self, record):
        # 精准屏蔽指定警告信息，不影响其他日志
        forbidden_message = "Caching is incompatible with gradient checkpointing in Qwen3DecoderLayer"
        return forbidden_message not in record.getMessage()

# 配置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 给transformers库添加过滤器（警告的来源）
transformers_logger = logging.getLogger("transformers")
transformers_logger.addFilter(QwenCacheWarningFilter())
transformers_logger.setLevel(logging.WARNING)  # 屏蔽冗余INFO日志

# 确保中文分词正常工作
try:
    import nltk
    nltk.data.path.append("/root/notebook/dataset")
except Exception as e:
    logger.warning(f"下载nltk punkt失败: {e}，可能影响中文分词效果")

# Qwen3-1.7B模型路径（固定配置）
QWEN3_1_7B_PATH = "/root/notebook/qinglan/model/Qwen/Qwen3-1.7B"

def parse_args():
    parser = argparse.ArgumentParser(description="基于Qwen3-1.7B微调GPU知识问答模型（全量数据训练）")
    parser.add_argument("--dataset_path", type=str, default="data.csv", 
                        help="GPU问答数据集路径（csv格式）")
    parser.add_argument("--output_dir", type=str, default="model_outputs", 
                        help="微调模型的输出目录")
    parser.add_argument("--lora_rank", type=int, default=8, 
                        help="LoRA适配器的秩")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="学习率")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, 
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="梯度累积步数")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="最大训练步数")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                        help="最大序列长度")
    parser.add_argument("--eval_steps", type=int, default=50, 
                        help="评估频率（若无需评估可忽略）")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="保存模型频率")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="是否使用4位量化加载模型（节省显存）")
    parser.add_argument("--do_train", action="store_true", 
                        help="是否进行训练")
    parser.add_argument("--do_eval", action="store_true", 
                        help="是否进行评估（若开启，仍使用全量数据做评估参考）")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="从检查点恢复训练")
    return parser.parse_args()

def load_gpu_qa_dataset(dataset_path):
    """加载GPU问答CSV数据集（全量数据作为训练集，无划分）"""
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集文件不存在: {dataset_path}")
    
    # 读取CSV文件（指定"问题"和"答案"列）
    df = pd.read_csv(dataset_path, encoding="utf-8")
    required_columns = ["问题", "答案"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要列: {col}，当前列名: {df.columns.tolist()}")
    
    # 去除空值样本
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    logger.info(f"读取CSV数据集成功，全量样本数: {len(df)}（全部用于训练）")
    
    # 转换为Hugging Face Dataset格式（直接返回全量数据集）
    full_dataset = Dataset.from_pandas(df)
    
    return full_dataset

def preprocess_gpu_qa(examples, tokenizer, max_length=1024):
    """预处理GPU问答数据集，转换为Qwen3-1.7B可接受的格式"""
    prompts = []
    
    for question, answer in zip(examples["问题"], examples["答案"]):
        # 构建GPU指令模板（适配大模型生成习惯）
        prompt = f"""### 问题:
{question}

### 回答:
{answer}"""
        prompts.append(prompt)
    
    # 编码文本（适配Qwen分词器）
    model_inputs = tokenizer(prompts, max_length=max_length, truncation=True, padding=False)
    
    # 准备标签（与输入相同，因果语言模型训练方式）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def create_qwen3_model_and_tokenizer(load_in_4bit=False):
    """创建Qwen3-1.7B模型和分词器（彻底禁用梯度检查点+修复量化配置）"""
    # ========== 修复load_in_4bit弃用警告 ==========
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        logger.info("已配置4位量化参数（BitsAndBytesConfig）")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN3_1_7B_PATH,
        local_files_only=True,
        padding_side="right",
        trust_remote_code=True
    )
    
    # 确保分词器有pad_token（Qwen模型默认可能无pad_token，使用eos_token替代）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Qwen分词器无pad_token，已使用eos_token替代")
    
    # 加载Qwen3-1.7B模型（移除直接传load_in_4bit，改用quantization_config）
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_1_7B_PATH,
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        quantization_config=quantization_config,  # 传入量化配置
        device_map="auto",  # 自动分配设备（GPU优先）
        local_files_only=True,
        trust_remote_code=True,
    )

    # ========== 彻底禁用梯度检查点（训练+评估阶段均生效） ==========
    model.gradient_checkpointing_disable()
    # 额外清除梯度检查点相关配置，防止隐式启用
    if hasattr(model, "config"):
        model.config.gradient_checkpointing = False
    logger.info("已彻底禁用梯度检查点（模型+配置层面），避免与缓存机制冲突")
    
    # 准备模型进行4位量化训练
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        # 量化后再次确认禁用梯度检查点
        model.gradient_checkpointing_disable()
        logger.info("已启用4位量化，模型已准备好进行低精度训练")
    
    logger.info(f"成功加载Qwen3-1.7B模型和分词器")
    return model, tokenizer

def setup_lora_config(rank=8):
    """设置LoRA配置（适配Qwen3-1.7B因果语言模型）"""
    return LoraConfig(
        r=rank,
        lora_alpha=16,  # alpha = 2*rank，常规配置
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",  # 因果语言模型任务
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Qwen模型关键模块
    )

def train_model(model, tokenizer, train_dataset, args):
    """训练Qwen3-1.7B模型（无验证集，全量数据训练）"""
    # 设置LoRA配置
    lora_config = setup_lora_config(args.lora_rank)

    # 统一用LoRA包装模型，确保拥有print_trainable_parameters方法
    model = get_peft_model(model, lora_config)
    # LoRA包装后再次禁用梯度检查点，防止包装过程中隐式启用
    model.gradient_checkpointing_disable()
    logger.info("已使用LoRA包装Qwen3-1.7B模型，支持可训练参数打印")
    
    # 打印可训练参数（查看LoRA参数量占比）
    model.print_trainable_parameters()
    
    # 设置训练参数（无验证集时，eval_strategy设为no）
    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=50,  # 预热步数，稳定训练
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_steps=None,  # 无验证集，关闭评估步数
        save_steps=args.save_steps,
        optim="paged_adamw_8bit",  # 高效优化器，节省显存
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # 余弦学习率衰减
        seed=42,  # 固定种子，可复现
        output_dir=args.output_dir,
        report_to="none",
        eval_strategy="no",  # 无验证集，关闭评估策略
        save_strategy="steps",
        load_best_model_at_end=False,  # 无验证集，无需加载最佳模型
        gradient_checkpointing=False,  # 训练参数层面再次禁用，双重保障
    )
    
    # 创建SFT训练器（无eval_dataset参数）
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        args=training_args,
        peft_config=lora_config,
    )
    
    # 开始训练
    logger.info("开始训练Qwen3-1.7B GPU问答模型（全量数据训练）...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存模型
    logger.info(f"训练完成，保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    return model

def evaluate_model(model, tokenizer, eval_dataset, args):
    """评估Qwen3-1.7B模型性能（ROUGE+BLEU指标，使用全量数据参考）"""
    logger.info("开始评估Qwen3-1.7B GPU问答模型...")
    
    # 加载评估指标
    rouge = Rouge()
    bleu_smoothing = SmoothingFunction().method4  # 平滑函数，避免BLEU为0
    
    # 准备评估结果列表
    results = []
    
    # 批量评估样本
    for example in eval_dataset:
        question = example["问题"]
        reference_answer = example["答案"]
        
        # 构建输入提示（与训练模板一致，确保生成效果）
        prompt = f"""### 问题:
{question}

### 回答:"""
        
        # 编码输入（转移到模型设备，添加truncation防止长度超标）
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
        
        # 生成回答（使用max_new_tokens避免截断，同时稳定生成参数）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,  # 简化传入方式，避免参数遗漏
                max_new_tokens=512,  # 替代max_length，避免输入+生成长度超标
                temperature=0.5,  # 适度增加随机性，避免重复
                top_p=0.9,
                top_k=40,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # 显式启用缓存，减少冲突概率
            )
        
        # 解码生成的回答并去除提示部分
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated_answer.replace(prompt, "").strip()
        
        # 过滤无关格式，让回答更干净
        filter_words = ["Human:", "Assistant:", "### 问题:", "### 回答:"]
        for word in filter_words:
            generated_answer = generated_answer.replace(word, "").strip()
        
        # 计算评估指标
        try:
            # ROUGE分数（中文文本适配）
            rouge_scores = rouge.get_scores(generated_answer, reference_answer)[0]
            
            # BLEU分数（中文字符级分词）
            gen_tokens = [char for char in generated_answer]
            ref_tokens = [char for char in reference_answer]
            
            bleu_score = sentence_bleu(
                [ref_tokens], 
                gen_tokens,
                smoothing_function=bleu_smoothing
            )
            
            # 保存单样本结果
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"],
                "bleu": bleu_score
            })
        except Exception as e:
            logger.error(f"评估样本出错（问题：{question[:20]}...）: {e}")
            # 出错时指标置0
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge-1": 0,
                "rouge-2": 0,
                "rouge-l": 0,
                "bleu": 0
            })
    
    # 计算平均指标
    avg_rouge_1 = np.mean([r["rouge-1"] for r in results])
    avg_rouge_2 = np.mean([r["rouge-2"] for r in results])
    avg_rouge_l = np.mean([r["rouge-l"] for r in results])
    avg_bleu = np.mean([r["bleu"] for r in results])
    
    # 打印评估结果
    logger.info(f"===== 评估结果汇总（全量数据参考） =====")
    logger.info(f"  平均ROUGE-1: {avg_rouge_1:.4f}")
    logger.info(f"  平均ROUGE-2: {avg_rouge_2:.4f}")
    logger.info(f"  平均ROUGE-L: {avg_rouge_l:.4f}")
    logger.info(f"  平均BLEU: {avg_bleu:.4f}")
    
    # 保存详细结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "gpu_evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细评估结果已保存到 {results_file}")
    
    return {
        "rouge-1": avg_rouge_1,
        "rouge-2": avg_rouge_2,
        "rouge-l": avg_rouge_l,
        "bleu": avg_bleu
    }

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录训练参数
    params_file = os.path.join(args.output_dir, "training_params.json")
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    logger.info(f"训练参数已保存到 {params_file}")
    
    # 1. 创建Qwen3-1.7B模型和分词器
    model, tokenizer = create_qwen3_model_and_tokenizer(args.load_in_4bit)
    
    # 2. 加载全量GPU问答数据集（无划分）
    full_dataset = load_gpu_qa_dataset(args.dataset_path)
    
    # 3. 预处理数据集并训练
    if args.do_train:
        # 预处理全量数据集（作为训练集）
        train_dataset = full_dataset.map(
            lambda x: preprocess_gpu_qa(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=full_dataset.column_names,
            desc="预处理全量训练集"
        )
        
        # 4. 训练模型（无验证集传入）
        model = train_model(model, tokenizer, train_dataset, args)
    
    # 5. 评估模型（若开启，使用全量原始数据做评估）
    if args.do_eval:
        # 如果未训练，加载微调后的LoRA模型
        if not args.do_train:
            # 先重新加载原始模型
            base_model, tokenizer = create_qwen3_model_and_tokenizer(args.load_in_4bit)
            model = PeftModel.from_pretrained(
                base_model, 
                args.output_dir if args.resume_from_checkpoint is None else args.resume_from_checkpoint
            )
            # 加载后再次禁用梯度检查点
            model.gradient_checkpointing_disable()
            logger.info(f"已加载预训练LoRA模型: {args.output_dir if args.resume_from_checkpoint is None else args.resume_from_checkpoint}")
        
        # 使用全量原始数据集进行评估
        evaluate_model(model, tokenizer, full_dataset, args)

if __name__ == "__main__":
    main()