import json
import nltk
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 导入 tqdm 进度条库



with open('./data/2qa/hotpotqa.json', 'r', encoding='utf-8') as f:
        ds = json.load(f)

def calculate_embeddings(content, model_path, vector_store_path):
    # 设置使用第二张 GPU（索引为1）
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
def vector_search(question):
    feature = emb_model.encode([question])
    # 粗粒度查询1个chunks
    distance, match_id = vector.search(feature, 2)
    # 获取chunks
    content = [chunk_data[int(i)] for i in match_id[0]]
    # 返回chunks和对应的id值 用于映射p
    return content, list(match_id[0])
# 用来存储块的原始文本和 ID
chunks_data = []  # 存储每个句子块的原始文本和块 ID
print('加载json完毕')
# 处理每个 JSON 数据条目
h=0
for entry in ds:
    context = entry['context']
    
    # 使用 NLTK 进行句子分割
    sentences = nltk.sent_tokenize(context)
    
    # 设置分块参数
    chunk_size = 4  # 每个块包含 4 个句子
    overlap = 2  # 前后重叠 2 个句子
    step = chunk_size - overlap  # 滑动窗口步长

    # 创建重叠块
    for i in range(0, len(sentences) - overlap, step):
        chunk = " ".join(sentences[i:i + chunk_size])  # 拼接4个句子作为一个块
        # 将原始块内容和对应 ID 保存到 chunks_data 中
        chunks_data.append({
            "id": entry["question_id"],  # 关联到原始文档 ID
            "chunk_index": h,  # 当前块的索引
            "text": chunk  # 当前块的文本
        })
        
        h += 1  # 递增索引
r=[]
for d in chunks_data:
     r.append(d['text'])
# 将文本块保存为 JSON 文件
with open('./data/corpus/processed/hotpotqa/chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=4)
calculate_embeddings(r,'../../LongRAG-main/multilingual-e5-large','./data/corpus/processed/hotpotqa/vector.index')
print('vector已生成.....\n')


# # 查询功能：基于一个问题找到最匹配的块
# def query_match(query, index, top_k=5):
#     # 将查询转换为向量
#     query_embedding = model.encode(query)
#     query_embedding = np.array([query_embedding], dtype=np.float32)

#     # 使用 FAISS 查找最接近的向量
#     distances, indices = index.search(query_embedding, top_k)

#     # 返回最接近的文本块
#     result = []
#     for i in range(top_k):
#         result.append({
#             "index": indices[0][i],  # FAISS 返回的索引
#             "distance": distances[0][i]  # 距离（相似度）
#         })
#     return result

# # 从 chunks_data.json 文件中读取块并返回对应文本
# def get_chunk_from_index(index, chunks_file='chunks_data.json'):
#     # 读取存储块信息的 JSON 文件
#     with open(chunks_file, 'r', encoding='utf-8') as f:
#         chunks_data = json.load(f)
    
#     # 从 FAISS 返回的索引中获取对应的文本
#     results = []
#     for idx in index:
#         chunk_info = chunks_data[idx]
#         results.append({
#             "text": chunk_info["text"],
#             "id": chunk_info["id"],
#             "chunk_index": chunk_info["chunk_index"]
#         })
#     return results

# # 示例查询
# query = "What was the mandatory crop in the French colonies?"
# results = query_match(query, index, top_k=3)

# # 输出查询结果
# for result in results:
#     chunk_index = result['index']
#     print(f"Match (Index {chunk_index}):")
#     chunks = get_chunk_from_index([chunk_index])
#     for chunk in chunks:
#         print(f"Text: {chunk['text']}\nID: {chunk['id']}\n")


# model_path='../../LongRAG-main/multilingual-e5-large'
# index_path = f'./data/corpus/processed/squad_qa/vector.index' # Vector index path
# emb_model = SentenceTransformer(model_path)
# vector = faiss.read_index(index_path)
# with open(f"./data/corpus/processed/squad_qa/chunks.json", "r") as fin:
#     chunk_data = json.load(fin)
# content,list=vector_search('Who are anthropologists working with along with other social scientists?')
# print(f'content0:{content[0]}')
# print(f'content1:{content[1]}')
