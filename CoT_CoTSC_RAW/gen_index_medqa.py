import os
import nltk
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 进度条库

# 确保 NLTK 资源可用
# nltk.download('punkt')

# 定义输入和输出文件
input_file = "./data/corpus/raw/merged.txt"  # 这里是整合后的大文本文件
output_faiss = "./data/corpus/processed/med/vector.index"
output_chunks = "./data/corpus/processed/med/chunks.json"

# 分块参数
chunk_size = 15  # 每个块包含 15 个句子
overlap = 2  # 前后重叠 2 个句子
step = chunk_size - overlap  # 滑动窗口步长

# 初始化句子编码器和 FAISS 索引
model = SentenceTransformer("../../LongRAG-main/multilingual-e5-large")
d = 1024  # 句子嵌入维度
index = faiss.IndexFlatIP(d)  # 使用点积索引（IP）
chunks_data = []
h = 0  # 分块索引

# 读取文本文件并进行分块
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()
    sentences = nltk.sent_tokenize(text)

    for i in tqdm(range(0, len(sentences) - overlap, step), desc="Encoding Chunks"):
        chunk = " ".join(sentences[i:i + chunk_size])
        embedding = model.encode([chunk])  # 生成文本块的向量
        embedding = np.array(embedding, dtype="float32")
        faiss.normalize_L2(embedding)  # 归一化向量，适用于 IP（点积）
        index.add(embedding)  # 添加到 FAISS 索引
        
        chunks_data.append({
            "chunk_index": h,  # 当前块的索引
            "text": chunk  # 当前块的文本
        })
        h += 1

# 保存分块后的数据
import json
with open(output_chunks, "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, indent=4)

print('已成功保存分块数据。')
# 保存 FAISS 索引
faiss.write_index(index, output_faiss)
print("已完成文本分块并构建 FAISS 索引")
