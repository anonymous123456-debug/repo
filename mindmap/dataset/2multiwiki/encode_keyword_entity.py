import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# 加载实体数据
entity_csv_path = "./entities.csv"  # 修改为你的实体CSV文件路径
entities_df = pd.read_csv(entity_csv_path)
entities = entities_df['Entity'].tolist()  # 获取所有实体

# 加载关键词数据（从之前的keywords.json中读取）
with open("./keyword.json", "r") as f:
    keywords_data = json.load(f)


# 提取所有问题的关键词
keywords = set()
for item in keywords_data:
    question_keywords = item.get("keywords", "")
    kws = question_keywords.split(",")
    keywords.update(kws)

# 将关键词转换为列表
keywords = list(keywords)

# 加载SentenceTransformer模型
# 脱敏：原私有 embedding 模型路径已用“”替代；运行时通过 MINDMAP_EMBEDDING_MODEL_PATH 提供真实路径。
model = SentenceTransformer(os.environ.get("MINDMAP_EMBEDDING_MODEL_PATH", "“”"))
model.to("cuda")

# 编码实体
entity_embeddings = model.encode(entities, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entities,
    "embeddings": entity_embeddings,
}

# 将实体的嵌入向量保存为 pickle 文件
with open("entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)

# 编码关键词
keyword_embeddings = model.encode(keywords, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": keywords,
    "embeddings": keyword_embeddings,
}

# 将关键词的嵌入向量保存为 pickle 文件
with open("keyword_embeddings.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("Done!")
