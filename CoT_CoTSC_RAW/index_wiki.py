
from datasets import load_dataset

from tqdm import tqdm  # 进度条库
import json
import re
from collections import defaultdict
import faiss
import numpy as np
import time 
import torch
import math
from sentence_transformers import SentenceTransformer
def split_text_into_sentences(text):
    """使用正则表达式将文本按句子分割"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # 以. ! ? 结尾的句子
    return [s.strip() for s in sentences if s.strip()]  # 去除空行

def chunk_sentences(sentences, chunk_size=20, overlap=2):
    """将句子列表按指定大小分块，并且前后块重叠一定数量的句子"""
    chunks = []
    start = 0
    while start < len(sentences):
        chunk = sentences[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))  # 将句子拼接成文本块
        start += (chunk_size - overlap)  # 移动窗口，保证重叠
    return chunks

# def calculate_embeddings(content, model_path, vector_store_path):
#     model = SentenceTransformer(model_path)
#     with tqdm(total=len(content), desc="Encoding sentences") as pbar:
#         content = [sentence if sentence is not None else '' for sentence in content]
#         embeddings = model.encode(content, convert_to_tensor=True,
#                                   show_progress_bar=False)  # Disable internal progress bar
#         for i in range(len(content)):
#             pbar.update(1)  # 手动更新进度条
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)  # 使用内积索引
#     index.add(embeddings)
#     faiss.write_index(index, vector_store_path)


def calculate_embeddings(content, model_path, vector_store_path, batch_size=1024):
    """逐批计算嵌入，防止 GPU OOM"""
    
    # **1. 加载模型并确保运行在 GPU**
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    print(device)

    # 检查模型参数所在的设备
    print("模型参数设备:", next(model.parameters()).device)

    # **2. 初始化进度条 & 预分配空间**
    embeddings_list = []
    with tqdm(total=len(content), desc="Encoding sentences") as pbar:
        
        # **3. 分批处理，防止 GPU OOM**
        for i in range(0, len(content), batch_size):
            batch = content[i:i + batch_size]
            batch = [sentence if sentence is not None else '' for sentence in batch]

            # **4. 计算嵌入**
            t1=time.time()
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            t2=time.time()
            print(f'time:{(t2-t1):.2f}s\n')
            # 打印返回张量的设备
            print("嵌入张量设备:", batch_embeddings.device)
            # **5. 移到 CPU，转换为 numpy**
            # batch_embeddings = batch_embeddings.cpu().numpy().astype("float32")
            embeddings_list.append(batch_embeddings)

            # **6. 更新进度条**
            pbar.update(len(batch))

    # **7. 合并所有批次的向量**
    # 在 GPU 上拼接所有结果，然后一次性转移到 CPU
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.cpu().numpy().astype("float32")
    print(f"✅ 生成 {embeddings.shape[0]} 个嵌入，向量维度: {embeddings.shape[1]}")
    # **7. 清理内存**
    del embeddings_list
    import gc
    gc.collect()
    # **8. 构建 FAISS 索引**
    # 假设我们有 100000 个 1024 维的嵌入向量
    L=len(content)
    nlist = nlist = max(1024, L // 100)       # 聚类数（推荐 √N 或 N/100）
    m = 32           # PQ 子向量数量（维度必须能被 m 整除）
    nbits = 8          # 量化位数（一般为 8-bit）
    dimension = embeddings.shape[1]
# 1️⃣ 先创建 `IndexFlatIP` 量化器（用于聚类）
    # quantizer = faiss.IndexFlatIP(dimension)
    quantizer= faiss.IndexFlatL2(dimension)  

# 2️⃣ 创建 `IndexIVFPQ`（压缩索引）
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

    faiss.normalize_L2(embeddings)  # 归一化整个数据集
# 3️⃣ 训练索引（必须先训练，再添加向量）
    print('开始训练')
    train_size = min(1000000, len(embeddings))  # 限制训练样本数
    print('结束训练')
    train_samples = embeddings[np.random.choice(len(embeddings), train_size, replace=False)]
    index.train(train_samples)

# 4️⃣ 添加向量到索引
    index.add(embeddings)

# 5️⃣ 保存索引到磁盘
    print(f"索引大小约为: {L * dimension * nbits / (8 * m) / 1e6:.2f} MB")
    faiss.write_index(index, vector_store_path)


save_path = f'./data/corpus/processed/wiki'
vector_store_path = f"{save_path}/vector.index"
# 读取 Wikipedia文件并处理
dataset = load_dataset("./data/wikidata", split="train")
output_chunks = []  # 存储所有分块
mapping_table = defaultdict(list)  # 记录文章 id 到分块 id 的映射

# 遍历 dataset 时显示进度条
# for data in tqdm(dataset, desc="Processing Wikipedia Articles"):
#     text_id = data["id"]
#     text = data["text"]
#     sentences = split_text_into_sentences(text)
#     chunks = chunk_sentences(sentences, chunk_size=20, overlap=2)

#     # 存储分块并建立映射
#     for i, chunk in enumerate(chunks):
#         chunk_id = f"{i}"  # 生成唯一块 ID
#         output_chunks.append({"chunk_id": chunk_id, "text": chunk})
#         mapping_table[chunk_id]=text_id

with open("./data/corpus/processed/wiki/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line.strip())  # 解析 JSON 行
        chunk =chunk['text']
        output_chunks.append(chunk)
# 示例：查看前 5 个数据
print(output_chunks[:5])

# # 保存分块数据
# with open("./data/corpus/processed/wiki/chunks.jsonl", "w", encoding="utf-8") as f:
#     for chunk in output_chunks:
#         f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

# # 保存映射表
# with open("./data/corpus/processed/wiki/id_to_rawid.json", "w", encoding="utf-8") as f:
#     json.dump(mapping_table, f, ensure_ascii=False, indent=4)
# print(f"✅ 处理完成！共生成 {len(output_chunks)} 个文本块。")

print("Calculating embeddings...")
start_time = time.time()
#嵌入模型还是使用llama3-8b进行实验
calculate_embeddings(output_chunks,'../../LongRAG-main/multilingual-e5-large', vector_store_path)
end_time = time.time()

print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
