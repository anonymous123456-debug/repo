import os

import re
import csv
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库显示进度条
# 加载 Llama 3 (用于 /llm 端点)
# 脱敏：原私有大模型路径已用“”替代；运行时通过 MINDMAP_LLM_MODEL_PATH 提供真实路径。
LLAMA_MODEL_PATH = os.environ.get("MINDMAP_LLM_MODEL_PATH", "“”")  # 本地模型路径或 Hugging Face 模型名称
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True)
llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

def generate_response(prompt, model, tokenizer,):
    messages =[{"role": "user", "content": prompt}]
    text = llama_tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = llama_tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = llama_model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=llama_tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response_text = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response_text

def extract_entities(text, model, tokenizer):
    prompt = f"""
    Extract all the entities from the following text and concatenate the extracted entities with ","

    text: {text}

    Output format: Entity 1, Entity 2,....
    """
    response = generate_response(prompt, model, tokenizer)
    entities = re.findall(r"([^,]+)", response)
    return entities
    
def extract_relations_based_on_entities(text, entities, model, tokenizer):
    """
    根据已提取的实体列表，提取实体之间的关系
    """
    # 构建提示符，限定关系提取的上下文
    entities_str = ", ".join(entities)
    prompt = f"""
    The following entities are present: {entities_str}.
    Extract the relationships between these entities from the following text:
    
    text: {text}

    Output format: (Entity 1, Relation, Entity 2)
    """
    response = generate_response(prompt, model, tokenizer)
    
    # 提取关系三元组
    relations = re.findall(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", response)
    return relations

def save_to_csv(entities, relations, entity_file, relation_file):
    # 保存实体
    with open(entity_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "Entity"])  # 添加索引列
        for idx, entity in enumerate(entities, start=1):  # 为实体添加索引
            writer.writerow([idx, entity])

    # 保存关系
    with open(relation_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Entity1", "Relation", "Entity2"])
        writer.writerows(relations)


# 文本分块函数：每块最多1200个词
def split_text_into_chunks(text, max_words=1200):
    words = text.split()  # 按空格分词
    chunks = []

    # 分割为每块最多1200词
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    return chunks

if __name__ == "__main__":
     # 加载文本文件
    with open('./commonsense.txt', "r", encoding="utf-8") as file:
        text = file.read()
    
    # 将文本按1200词分块
    chunks = split_text_into_chunks(text, max_words=1200)
    
    # 初始化存储提取的实体和关系
    all_entities = []
    all_relations = []
    
    # 使用 tqdm 显示进度条
    overall_progress = tqdm(total=len(chunks), desc="Overall Progress", unit="task")
    for chunk in chunks:
        # 第一步：提取实体
        entities = extract_entities(chunk, llama_model, llama_tokenizer)
        all_entities.extend(entities)

        # 第二步：根据提取的实体，提取关系
        relations = extract_relations_based_on_entities(chunk, entities, llama_model, llama_tokenizer)
        all_relations.extend(relations)
        overall_progress.update(1)
    overall_progress.close()
    # 保存提取结果到 CSV
    save_to_csv(all_entities, all_relations, "entities_deepseek.csv", "relations_deepseek.csv")
    print("Processing complete!")
