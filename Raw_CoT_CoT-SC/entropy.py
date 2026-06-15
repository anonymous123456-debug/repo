import os
import argparse
import json
import numpy as np
import torch
import random
import transformers
from tqdm import tqdm
import time 
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设你的 metric.py 在路径中，包含这些函数
# from metric import compute_average_metrics, F1_scorer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_attention_entropy(attentions):
    """
    计算生成最后一个token时，最后一层所有Head的平均注意力熵
    attentions: tuple (num_layers), 每个元素为 [batch, num_heads, 1, seq_len]
    """
    # 取最后一层
    last_layer_attn = attentions[-1] 
    # 取最后一个生成的token对前面所有token的分布 [heads, 1, seq_len]
    attn_weights = last_layer_attn[0, :, -1, :] 
    
    epsilon = 1e-12
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + epsilon), dim=-1)
    return torch.mean(entropy).item()

def load_ds(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    return [{
        'id': d['question_id'],
        'context': d['context'],
        'answer': d['answer'],
        'question': d['question']
    } for d in ds]

def run_step_stress_test(model, tokenizer, dataset, steps_limit):
    """
    针对特定步数限制运行测试
    """
    predictions = []
    ground_truths = []
    entropies = []
    
    print(f"\n>>> 正在测试强制推理步数: {steps_limit}步")
    
    for q in tqdm(dataset, desc=f"Steps {steps_limit}"):
        # 强制 Prompt：要求模型必须严格按照指定的步数进行逻辑拆解
        prompt = f"""Given the context, answer the question step by step. 
You MUST use exactly {steps_limit} reasoning steps before giving the final answer.
Format:
Reasoning:
1. ...
2. ...
...
{steps_limit}. ...
Final Answer: [Your short answer]

Context: {q['context']}
Question: {q['question']}
Reasoning:
1."""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 统计推理开始时的输入长度
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                output_attentions=True,       # 关键：输出注意力权重
                return_dict_in_generate=True, # 关键：返回字典格式
                pad_token_id=tokenizer.eos_token_id
            )

        # 1. 提取文本响应
        generated_tokens = outputs.sequences[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 2. 提取最后一行作为答案
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        last_line = lines[-1] if lines else ""
        pred_answer = last_line.replace("Final Answer:", "").replace("final answer:", "").strip().lower()
        if pred_answer.endswith('.'): pred_answer = pred_answer[:-1]

        predictions.append(pred_answer)
        ground_truths.append(q['answer'].lower())

        # 3. 计算生成结束时的注意力熵（反映模型此时的“疲劳”或“失焦”程度）
        # outputs.attentions 是一个元组，长度等于生成的token数
        if outputs.attentions:
            last_token_attentions = outputs.attentions[-1] # 最后一个token生成的注意力
            entropy = calculate_attention_entropy(last_token_attentions)
            entropies.append(entropy)

    # 简易版准确率计算（如果你没有导入外部metric，这里做简单匹配）
    correct = sum([1 if p == g else 0 for p, g in zip(predictions, ground_truths)])
    em = (correct / len(dataset)) * 100
    avg_entropy = np.mean(entropies) if entropies else 0
    
    return em, avg_entropy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"), help="Local model path or Hugging Face model id")
    args = parser.parse_args()

    seed_everything(931)

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map='auto'
    )

    ds = load_ds('./data/2qa/2WikiMultiHopQA.json')[:200]

    step_configs = [5, 10, 15, 20]
    results_table = []

    for s in step_configs:
        em_score, avg_entropy = run_step_stress_test(model, tokenizer, ds, s)
        results_table.append({
            "steps": s,
            "em": em_score,
            "entropy": avg_entropy
        })

    # 打印实验报告
    print("\n" + "="*50)
    print(f"{'推理步数':<10} | {'准确率 (EM)':<15} | {'平均注意力熵 (Entropy)'}")
    print("-" * 50)
    for res in results_table:
        print(f"{res['steps']:<12} | {res['em']:<15.2f} | {res['entropy']:.4f}")
    print("="*50)
