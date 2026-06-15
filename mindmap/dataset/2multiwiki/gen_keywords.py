import os
import re
import csv
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm  # 导入 tqdm 库显示进度条
# 加载 Llama 3 (用于 /llm 端点)
# 脱敏：原私有大模型路径已用“”替代；运行时通过 MINDMAP_LLM_MODEL_PATH 提供真实路径。
LLAMA_MODEL_PATH = os.environ.get("MINDMAP_LLM_MODEL_PATH", "“”")  # 本地模型路径或 Hugging Face 模型名称
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True)
llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

def generate_response(prompt, model, tokenizer):
    messages =[{"role": "user", "content": prompt}]
    text = llama_tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = llama_tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = llama_model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=100,pad_token_id=llama_tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response_text = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response_text

def process_string(s: str) -> str:
    # 提取冒号后面的字符串
    if ':' in s:
        s = s.split(':', 1)[1].strip()  # 只分割一次，取冒号后面的部分
    
    # 去掉除了逗号以外的所有标点符号
    s = re.sub(r'[^\w\s,]', '', s)
    keywords = [keyword.strip() for keyword in s.split(',')]
    cleaned_string = ",".join(keywords)
    return cleaned_string
def extract_keywords(text):
    # prompt = f"""
    # Extract all the keywords from the following question and concatenate the extracted entities with ","

    # text: {text}

    # Output format: keywords 1, keywords 2,....
    # """
    prompt = f"Extract keywords from the following question, separated by commas: {text}"
    keywords=generate_response(prompt,llama_model,llama_tokenizer)
    keywords=process_string(keywords)
    print(f'keywords:{keywords}\n')
    return keywords

with open('./data/2WikiMultiHopQA.json', 'r', encoding='utf-8') as f:
        data = json.load(f)[:200]
result = []
overall_progress = tqdm(total=len(data), desc="Overall Progress", unit="task")
for item in data:
    question = item["question"]
    keywords = extract_keywords(question)
    result.append({
        "id": item["question_id"],
        "question": question,
        "keywords": keywords,
        "context": item["context"],
        "answer": item["answer"]
    })
    overall_progress.update(1)
overall_progress.close()

# 保存结果到 keyword.json
with open("keyword.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("Keyword extraction completed and saved to keyword.json")
