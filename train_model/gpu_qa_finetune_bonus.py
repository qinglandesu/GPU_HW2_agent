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
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraConfig
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# ========== 第一步：添加精准日志过滤器（核心，彻底屏蔽目标警告） ==========
import logging
import warnings
warnings.filterwarnings("ignore", message="Caching is incompatible with gradient checkpointing in Qwen3DecoderLayer")

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

# Qwen3-1.7B模型路径（固定配置）
QWEN3_1_7B_PATH = "/root/notebook/qinglan/model/Qwen/Qwen3-1.7B"

def parse_args():
    parser = argparse.ArgumentParser(description="基于Qwen3-1.7B微调GPU知识问答模型（全量数据训练）")
    parser.add_argument("--dataset_path", type=str, default="processed_dataset1.jsonl", 
                        help="GPU问答数据集路径（jsonl格式）")
    parser.add_argument("--output_dir", type=str, default="model_outputs", 
                        help="微调模型的输出目录")
    parser.add_argument("--lora_rank", type=int, default=64,  # 根据数据集说明修改为64
                        help="LoRA适配器的秩")
    parser.add_argument("--lora_alpha", type=int, default=128,  # 根据数据集说明添加缩放因子
                        help="LoRA的缩放因子")
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
    parser.add_argument("--test_size", type=float, default=0.1, 
                        help="测试集比例（0-1之间）")
    return parser.parse_args()

def load_gpu_qa_dataset(dataset_path, test_size=0.1):
    """加载GPU问答JSONL数据集"""
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集文件不存在: {dataset_path}")
    
    # 读取JSONL文件
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    # 提取instruction, question, answer字段
                    instruction = item.get('instruction', '')
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    
                    # 根据数据集说明：instruction对应system_promt，question对应input，answer对应output
                    # 我们可以合并instruction和question作为输入
                    if instruction and instruction not in question:
                        full_question = f"{instruction}\n{question}"
                    else:
                        full_question = question
                    
                    data.append({
                        "instruction": instruction,
                        "question": full_question,
                        "answer": answer
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSONL行时出错: {e}, 行内容: {line[:100]}...")
    
    if not data:
        raise ValueError(f"数据集文件为空或格式不正确: {dataset_path}")
    
    logger.info(f"读取JSONL数据集成功，总样本数: {len(data)}")
    
    # 转换为Hugging Face Dataset格式
    dataset = Dataset.from_list(data)
    
    # 分割数据集（如果指定了测试集比例）
    if test_size > 0 and test_size < 1:
        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"数据集分割: 训练集 {len(train_dataset)} 条, 测试集 {len(eval_dataset)} 条")
        return train_dataset, eval_dataset
    else:
        logger.info(f"使用全量数据作为训练集: {len(dataset)} 条")
        return dataset, None

def preprocess_gpu_qa(examples, tokenizer, max_length=1024):
    """预处理GPU问答数据集，转换为Qwen3-1.7B可接受的格式"""
    texts = []
    
    for question, answer in zip(examples["question"], examples["answer"]):
        # 构建对话格式（适配Qwen Chat格式）
        # 根据Qwen官方推荐格式：<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>
        
        # 提取instruction作为system prompt
        instruction = examples.get("instruction", [""] * len(examples["question"]))
        idx = len(texts)
        system_msg = instruction[idx] if idx < len(instruction) else ""
        
        # 构建完整对话
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
        
        # 使用tokenizer.apply_chat_template格式化
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # 编码文本
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    # 准备标签（与输入相同，因果语言模型训练方式）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def create_qwen3_model_and_tokenizer(load_in_4bit=False):
    """创建Qwen3-1.7B模型和分词器"""
    # 修复load_in_4bit弃用警告
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
    
    # 设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 填充到右侧
    
    # 加载Qwen3-1.7B模型
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_1_7B_PATH,
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        use_cache=True  # 启用缓存以提高速度
    )

    # 禁用梯度检查点
    model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.gradient_checkpointing = False
    logger.info("已彻底禁用梯度检查点")
    
    # 准备模型进行4位量化训练
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_disable()
        logger.info("已启用4位量化，模型已准备好进行低精度训练")
    
    logger.info(f"成功加载Qwen3-1.7B模型和分词器")
    return model, tokenizer

def setup_lora_config(rank=64, alpha=128):
    """设置LoRA配置（根据数据集建议参数）"""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """训练Qwen3-1.7B模型"""
    # 设置LoRA配置
    lora_config = setup_lora_config(args.lora_rank, args.lora_alpha)

    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_disable()
    logger.info("已使用LoRA包装Qwen3-1.7B模型")
    
    # 打印可训练参数
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"总参数: {all_params:,}")
    logger.info(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=50,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset is not None else False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )
    
    # 创建SFT训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            padding=True,
            return_tensors="pt"
        ),
        args=training_args,
        peft_config=lora_config,
    )
    
    # 开始训练
    logger.info("开始训练Qwen3-1.7B GPU问答模型...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存模型
    logger.info(f"训练完成，保存模型到 {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    return model

def evaluate_model(model, tokenizer, eval_dataset, args):
    """评估Qwen3-1.7B模型性能"""
    logger.info("开始评估Qwen3-1.7B GPU问答模型...")
    
    # 初始化评估指标
    rouge = Rouge()
    bleu_smoothing = SmoothingFunction().method4
    
    results = []
    
    # 评估样本
    for i, example in enumerate(eval_dataset):
        question = example["question"]
        instruction = example.get("instruction", "")
        reference_answer = example["answer"]
        
        # 构建输入消息
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        
        # 使用tokenizer.apply_chat_template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码生成结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取assistant的回答（去掉用户输入部分）
        generated_answer = generated_text.replace(prompt, "").strip()
        
        # 计算评估指标
        try:
            # ROUGE分数
            rouge_scores = rouge.get_scores(generated_answer, reference_answer)[0]
            
            # BLEU分数（字符级）
            gen_chars = list(generated_answer)
            ref_chars = list(reference_answer)
            
            bleu_score = sentence_bleu(
                [ref_chars],
                gen_chars,
                smoothing_function=bleu_smoothing
            )
            
            results.append({
                "instruction": instruction,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"],
                "bleu": bleu_score
            })
            
        except Exception as e:
            logger.warning(f"评估第{i}个样本出错: {e}")
            results.append({
                "instruction": instruction,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge-1": 0,
                "rouge-2": 0,
                "rouge-l": 0,
                "bleu": 0
            })
    
    # 计算平均指标
    if results:
        avg_rouge_1 = np.mean([r["rouge-1"] for r in results])
        avg_rouge_2 = np.mean([r["rouge-2"] for r in results])
        avg_rouge_l = np.mean([r["rouge-l"] for r in results])
        avg_bleu = np.mean([r["bleu"] for r in results])
        
        logger.info(f"===== 评估结果汇总 =====")
        logger.info(f"  平均ROUGE-1: {avg_rouge_1:.4f}")
        logger.info(f"  平均ROUGE-2: {avg_rouge_2:.4f}")
        logger.info(f"  平均ROUGE-L: {avg_rouge_l:.4f}")
        logger.info(f"  平均BLEU: {avg_bleu:.4f}")
        
        # 保存详细结果
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
    else:
        logger.warning("没有有效的评估结果")
        return None

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存训练参数
    params_file = os.path.join(args.output_dir, "training_params.json")
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    logger.info(f"训练参数已保存到 {params_file}")
    
    # 1. 加载数据集
    if args.do_eval and not args.do_train:
        # 如果只评估，加载完整数据集
        full_dataset, _ = load_gpu_qa_dataset(args.dataset_path, test_size=0)
    else:
        # 如果训练，分割数据集
        train_dataset, eval_dataset = load_gpu_qa_dataset(args.dataset_path, args.test_size)
    
    # 2. 创建模型和分词器
    model, tokenizer = create_qwen3_model_and_tokenizer(args.load_in_4bit)
    
    # 3. 预处理数据集
    if args.do_train:
        logger.info("预处理训练数据集...")
        processed_train_dataset = train_dataset.map(
            lambda x: preprocess_gpu_qa(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="预处理训练集"
        )
        
        if eval_dataset is not None:
            logger.info("预处理评估数据集...")
            processed_eval_dataset = eval_dataset.map(
                lambda x: preprocess_gpu_qa(x, tokenizer, args.max_seq_length),
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="预处理评估集"
            )
        else:
            processed_eval_dataset = None
        
        # 4. 训练模型
        model = train_model(model, tokenizer, processed_train_dataset, processed_eval_dataset, args)
    
    # 5. 评估模型
    if args.do_eval:
        if not args.do_train:
            # 加载已训练的模型
            base_model, tokenizer = create_qwen3_model_and_tokenizer(args.load_in_4bit)
            model = PeftModel.from_pretrained(
                base_model,
                args.output_dir if args.resume_from_checkpoint is None else args.resume_from_checkpoint
            )
            model.gradient_checkpointing_disable()
            model = model.merge_and_unload()  # 合并LoRA权重
            logger.info(f"已加载预训练模型")
        
        # 加载评估数据集
        if args.do_train and eval_dataset is not None:
            eval_data = eval_dataset
        else:
            full_dataset, _ = load_gpu_qa_dataset(args.dataset_path, test_size=0)
            # 取前100个样本评估（加快速度）
            if len(full_dataset) > 100:
                eval_data = full_dataset.select(range(100))
            else:
                eval_data = full_dataset
        
        evaluate_model(model, tokenizer, eval_data, args)

if __name__ == "__main__":
    main()