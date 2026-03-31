import json
import time 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def vector_search(question):
    t1=time.time()
    feature = emb_model.encode([question])
    print(f'vector{vector.ntotal}\n')
    # 粗粒度查询1个chunks
    distance, match_id = vector.search(feature, 5)
    match_id_set = set(int(i) for i in match_id[0])  # 转换为集合，方便快速查找
    print(match_id_set)
    content = []  # 用字典存储匹配的数据
    with open("./data/corpus/processed/wiki/chunks.jsonl", "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):  # 按行读取，避免内存溢出
            if idx==0:print(json.loads(line)['text'])
            if idx in match_id_set:  # 仅匹配需要的行
                content.append(json.loads(line)['text'])
                print('-----0--------\n')
                print(json.loads(line)['text'])
                print('------1-------\n')
            if len(content) == len(match_id_set):  # 读到所有匹配的块就提前结束
                break
    # 返回匹配的 chunks 和 ID 列表
    t2=time.time()
    tt=float(t2-t1)
    print(f'time:{tt:.2f} s\n')
    return content, list(match_id[0])
question="Anarchism is a political philosophy and movement that is skeptical of all justifications for authority and seeks to abolish the institutions it claims maintain unnecessary coercion and hierarchy, typically including nation-states, and capitalism. Anarchism advocates for the replacement of the state with stateless societies and voluntary free associations. As a historically left-wing movement, this reading of anarchism is placed on the farthest left of the political spectrum, usually described as the libertarian wing of the socialist movement (libertarian socialism). Humans have lived in societies without formal hierarchies long before the establishment of states, realms, or empires. With the rise of organised hierarchical bodies, scepticism toward authority also rose. Although traces of anarchist ideas are found all throughout history, modern anarchism emerged from the Enlightenment. During the latter half of the 19th and the first decades of the 20th century, the anarchist movement flourished in most parts of the world and had a significant role in workers' struggles for emancipation. Various anarchist schools of thought formed during this period. Anarchists have taken part in several revolutions, most notably in the Paris Commune, the Russian Civil War and the Spanish Civil War, whose end marked the end of the classical era of anarchism. In the last decades of the 20th and into the 21st century, the anarchist movement has been resurgent once more, growing in popularity and influence within"
model_path='../../LongRAG-main/multilingual-e5-large'
index_path='./data/corpus/processed/wiki/vector.index'
emb_model = SentenceTransformer(model_path)
vector = faiss.read_index(index_path)

content,idx_index=vector_search(question)
# arr=[]
# with open("./data/corpus/processed/wiki/chunks.jsonl", "r", encoding="utf-8") as fin:
#         for idx, line in enumerate(fin):  # 按行读取，避免内存溢出
#             arr.append(json.loads(line)['text'])
           