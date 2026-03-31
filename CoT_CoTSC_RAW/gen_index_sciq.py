import os
import nltk
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 进度条库
import json
# 确保 NLTK 资源可用
# nltk.download('punkt')

# 定义输入和输出文件
input_file = "./data/corpus/raw/sciq_corpus.json"  # 这里是整合后的大文本文件
output_faiss = "./data/corpus/processed/sciq/vector.index"
output_chunks = "./data/corpus/processed/sciq/chunks.json"



# 初始化句子编码器和 FAISS 索引
model = SentenceTransformer("../../LongRAG-main/multilingual-e5-large")
d = 1024  # 句子嵌入维度
index = faiss.IndexFlatIP(d)   # 使用点积索引（IP）


# 读取
with open(input_file, "r", encoding="utf-8") as f:
    support_data = json.load(f)

# 提取文本  
batch_size = 32  # 设置批次大小，避免一次性占用太多显存
texts = [entry["text"] for entry in support_data]
print(len(texts))
embeddings=[]
for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Texts"):
    batch_texts = texts[i:i + batch_size]  # 获取当前批次
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)  # 编码
    batch_embeddings = batch_embeddings.astype(np.float32)  # 转换数据格式
    faiss.normalize_L2(batch_embeddings)  # 归一化向量，适用于 IP（点积相似度）
    
    index.add(batch_embeddings)  # 添加到 FAISS 索引
    embeddings.append(batch_embeddings)


# 保存 FAISS 索引
faiss.write_index(index, output_faiss)


print("已完成文本分块并构建 FAISS 索引")
