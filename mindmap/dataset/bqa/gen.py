import re
import csv
import json
import subprocess
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import pickle
from datasets import load_dataset

# ========== 模型加载 ==========
# 脱敏：原私有大模型路径已用“”替代；运行时通过 MINDMAP_LLM_MODEL_PATH 提供真实路径。
LLAMA_MODEL_PATH = os.environ.get("MINDMAP_LLM_MODEL_PATH", "“”")
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = llama_tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = llama_model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.get('attention_mask'),
        max_new_tokens=4000,
        pad_token_id=llama_tokenizer.eos_token_id
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def process_string(s: str) -> str:
    if ':' in s:
        s = s.split(':', 1)[1].strip()
    s = re.sub(r'[^\w\s,]', '', s)
    keywords = [k.strip() for k in s.split(',')]
    return ",".join(keywords)

def extract_keywords(text):
    prompt = f"Extract keywords from the following question, separated by commas: {text}"
    keywords = generate_response(prompt)
    keywords = process_string(keywords)

    # 转成列表，去空格、去空值
    keywords_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    # 去重（保持原顺序）
    keywords_list = list(dict.fromkeys(keywords_list))
    # 取前 10 个
    keywords_list = keywords_list[:10]

    # 再拼成字符串
    return ",".join(keywords_list)

def extract_entities(text):
    prompt = f"""
    Extract all the entities from the following text and concatenate the extracted entities with ","

    text: {text}

    Output format: Entity 1, Entity 2,....
    """
    response = generate_response(prompt)
    entities = re.findall(r"([^,]+)", response)
    return [e.strip() for e in entities if e.strip()]

def extract_relations_based_on_entities(text, entities):
    entities_str = ", ".join(entities)
    prompt = f"""
    The following entities are present: {entities_str}.
    Extract the relationships between these entities from the following text:
    
    text: {text}

    Output format: (Entity 1, Relation, Entity 2)
    """
    response = generate_response(prompt)
    relations = re.findall(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", response)
    return [(e1.strip(), rel.strip(), e2.strip()) for e1, rel, e2 in relations]

def save_to_csv(entities, relations, entity_file, relation_file):
    with open(entity_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "Entity"])
        unique_entities = sorted(set(entities))
        for idx, entity in enumerate(unique_entities, start=1):
            writer.writerow([idx, entity])

    with open(relation_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Entity1", "Relation", "Entity2"])
        for row in relations:
            writer.writerow(row)

def split_text_into_chunks(text, max_words=1200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def filter_with_keywords(keyword_file, raw_entity_file, raw_relation_file, out_entity_file, out_relation_file):
    with open(keyword_file, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)
    keywords_set = set()
    for item in keywords_data:
        kws = item.get("keywords","").split(",")
        kws = [k.strip() for k in kws]
        keywords_set.update(kws)

    df_entities = pd.read_csv(raw_entity_file)
    filtered_entities = df_entities[df_entities['Entity'].isin(keywords_set)]

    df_relations = pd.read_csv(raw_relation_file)
    filtered_relations = df_relations[
        df_relations['Entity1'].isin(filtered_entities['Entity']) &
        df_relations['Entity2'].isin(filtered_entities['Entity'])
    ]

    filtered_entities.to_csv(out_entity_file, index=False, encoding='utf-8')
    filtered_relations.to_csv(out_relation_file, index=False, encoding='utf-8')

def encode_and_save_embeddings(entity_file, keyword_file, entity_emb_file, keyword_emb_file):
    model = SentenceTransformer(os.environ.get("MINDMAP_EMBEDDING_MODEL_PATH", "“”"))
    model.to("cuda")

    df_entities = pd.read_csv(entity_file)
    entities = df_entities['Entity'].tolist()
    entity_embeddings = model.encode(entities, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    with open(entity_emb_file, "wb") as f:
        pickle.dump({"entities": entities, "embeddings": entity_embeddings}, f)

    with open(keyword_file, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)
    keywords = []
    for item in keywords_data:
        kws = item.get("keywords", "")
        kws_list = [k.strip() for k in kws.split(",")]
        keywords.extend(kws_list)
    keywords = list(set(keywords))

    keyword_embeddings = model.encode(keywords, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    with open(keyword_emb_file, "wb") as f:
        pickle.dump({"keywords": keywords, "embeddings": keyword_embeddings}, f)

if __name__ == "__main__":
    # 1. 关键词抽取并保存 keyword.json
    # 1. 关键词抽取并保存 keyword.json
    data=[]
    with open('./data/bqa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:200]

    result = []
    overall_progress = tqdm(total=len(data), desc="Overall Progress keyword", unit="task")
    for item in data:
        question = item["question"]
        keywords = extract_keywords("question:"+question+"context:"+item['context'])
        result.append({
            "keywords": keywords,
            "context":item['context'],
            "answer": item["answer"],
            "options": [],
            "question":item['question']
        })
        overall_progress.update(1)
    overall_progress.close()

    with open("keyword.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("关键词抽取完成，已保存到 keyword.json")

    # 2. 实体关系抽取，保存 raw csv
    with open('./bqa.txt', "r", encoding="utf-8") as file:
        text = file.read()

    chunks = split_text_into_chunks(text, max_words=1200)
    all_entities = []
    all_relations = []

    for chunk in tqdm(chunks, desc="实体关系抽取"):
        entities = extract_entities(chunk)
        all_entities.extend(entities)
        relations = extract_relations_based_on_entities(chunk, entities)
        all_relations.extend(relations)

    # os.makedirs("output", exist_ok=True)
    raw_entity_file = "./raw_entities.csv"
    raw_relation_file = "./raw_relations.csv"
    save_to_csv(all_entities, all_relations, raw_entity_file, raw_relation_file)
    print("实体关系抽取完成，已保存到 CSV 文件")

    # 3. 筛选实体关系
    filtered_entity_file = "./entities.csv"
    filtered_relation_file = "./relation.csv"
    filter_with_keywords("keyword.json", raw_entity_file, raw_relation_file, filtered_entity_file, filtered_relation_file)

    # 4. 编码向量
    encode_and_save_embeddings(filtered_entity_file, "keyword.json", "./entity_embeddings.pkl", "./keyword_embeddings.pkl")

    print("全部流程执行完成！")
