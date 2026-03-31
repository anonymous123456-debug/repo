import json
import re
import os
import time
from sentence_transformers import SentenceTransformer
import faiss
import argparse
from tqdm import tqdm
import yaml

# 传递命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process dataset and generate embeddings.")
    parser.add_argument("--dataset", help="Name of the dataset", type=str)
    return parser.parse_args()

#计算给定文本的单词数量
def get_word_count(text):
    #匹配任何非单词字符（即除了字母、数字和下划线之外的任何字符）
    regEx = re.compile('[\W]')
    #匹配中文字符
    chinese_char_re = re.compile(r"([\u4e00-\u9fa5])")
    #用非单词字符分割成多个部分
    words = regEx.split(text.lower())
    word_list = []
    for word in words:
        if chinese_char_re.split(word):
            word_list.extend(chinese_char_re.split(word))
        else:
            word_list.append(word)
    return len([w for w in word_list if len(w.strip()) > 0])

# 去掉 split_sentences 函数，不再分块
def process_data(file_path, save_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_rawid = {}
    processed_chunks = []

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing data"):
        # 这里的每个content就是p
        content = item.get("text") or item.get("ch_content") or item.get("ch_contenn")
        
        # 不进行分块，直接将原文加入 processed_chunks
        processed_chunks.append(content)
        
        # 存储处理后数据块的索引与原始数据索引的对应关系。
        id_to_rawid[len(processed_chunks) - 1] = idx
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/chunks.json", "w", encoding='utf-8') as fout:
        json.dump(processed_chunks, fout, ensure_ascii=False)
    with open(f"{save_path}/id_to_rawid.json", "w", encoding='utf-8') as fout:
        json.dump(id_to_rawid, fout, ensure_ascii=False)
    
    return processed_chunks


def calculate_embeddings(content, model_path, vector_store_path):
    model = SentenceTransformer(model_path)
    with tqdm(total=len(content), desc="Encoding sentences") as pbar:
        content = [sentence if sentence is not None else '' for sentence in content]
        embeddings = model.encode(content, convert_to_tensor=True,
                                  show_progress_bar=False)  # Disable internal progress bar
        for i in range(len(content)):
            pbar.update(1)  # 手动更新进度条
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积索引
    index.add(embeddings)
    faiss.write_index(index, vector_store_path)


def main():
    args = parse_arguments()
    save_path = f'./data/corpus/processed/{args.dataset}'
    vector_store_path = f"{save_path}/vector.index"
    
    print("Starting data processing...")
    content = process_data(f"./data/corpus/raw/{args.dataset}.json", save_path)
    
    print("Calculating embeddings...")
    start_time = time.time()
    #嵌入模型还是使用llama3-8b进行实验
    calculate_embeddings(content,'../../LongRAG-main/multilingual-e5-large', vector_store_path)
    end_time = time.time()
    
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
