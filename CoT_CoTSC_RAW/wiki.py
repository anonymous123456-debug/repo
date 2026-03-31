

import faiss
from sentence_transformers import SentenceTransformer
import argparse
import json
import numpy as np
import torch
import random
import transformers
from tqdm import tqdm
from datasets import load_dataset
import time 
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,GPT2Tokenizer, GPT2Model
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,help="Name of the dataset")
parser.add_argument("--method", type=str,help="Name of the dataset")
args = parser.parse_args()

'''
加载模型
'''
modelname='../llama3-8b'
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
def vector_search(question):
    t1=time.time()
    feature = emb_model.encode([question])
    # 粗粒度查询1个chunks
    distance, match_id = vector.search(feature, 1)
    match_id_set = set(int(i) for i in match_id[0])  # 转换为集合，方便快速查找
    
    content = []  # 用字典存储匹配的数据
    with open("./data/corpus/processed/wiki/chunks.jsonl", "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):  # 按行读取，避免内存溢出
            if idx in match_id_set:  # 仅匹配需要的行
                content.append(json.loads(line)['text'])
            if len(content) == len(match_id_set):  # 读到所有匹配的块就提前结束
                break
    # 返回匹配的 chunks 和 ID 列表
    t2=time.time()
    tt=float(t2-t1)
    print(content)
    print(f'time:{tt:.2f} s\n')
    return content, list(match_id[0])

def raw_answer(results):
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        prompt="Assuming you are A know-it-all, I will now ask you some common sense questions and give you 5 possible answers to that question Options A,B,C,D, E. Please choose the best answer from these options. Please answer the question with a single letter option, without adding any words.\nQuestion:"+q['question']+"\nOptions:A."+q['choices']['text'][0]+"\nB."+q['choices']['text'][1]+"\nC."+q['choices']['text'][2]+"\nD."+q['choices']['text'][3]+"\nE."+q['choices']['text'][4]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if response==q["answerKey"]:
            correct_answer+=1
            correct_list.append(q['question'])
        else:
            wrong_list.append(q['question'])
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    print(f'time:{cost:.2f} s')
    print(f'total_answer:{total_answer}')
    print(f'correct_answer:{correct_answer}')
    print(f'Accuracy:{correct_answer/total_answer}')
    # 关闭总体进度条
    overall_progress.close()
        # 将正确和错误回答的问题保存到JSON文件
    with open(f'./analysis/raw/{args.dataset}_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/raw/{args.dataset}_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
def rag_answer(results):
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        content,list=vector_search(q['question'])
        prompt="Assuming you are A know-it-all, I will now ask you some common sense questions and give you 5 options A, B, C, D, E, and additional background on that question. Please think deeply about the questions and background knowledge, and choose the best answer from these options. Please answer this question with a letter option, without adding any words. \nQuestion:"+q['question']+"\nOptions:A."+q['choices']['text'][0]+"\nB."+q['choices']['text'][1]+"\nC."+q['choices']['text'][2]+"\nD."+q['choices']['text'][3]+"\nE."+q['choices']['text'][4]+"\nbackground Knowledge:"+content[0]
        print(f'prompt:{prompt}\n')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if response==q["answerKey"]:
            correct_answer+=1
            correct_list.append(q['question'])
        else:
            wrong_list.append(q['question'])
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    print(f'time:{cost:.2f} s')
    print(f'total_answer:{total_answer}')
    print(f'correct_answer:{correct_answer}')
    print(f'Accuracy:{correct_answer/total_answer}')
    # 关闭总体进度条
    overall_progress.close()
        # 将正确和错误回答的问题保存到JSON文件
    # with open(f'./analysis/rag/{args.dataset}_correct_answers.json', 'w', encoding='utf-8') as f:
    #     json.dump(correct_list, f, ensure_ascii=False, indent=4)

    # with open(f'./analysis/rag/{args.dataset}_wrong_answers.json', 'w', encoding='utf-8') as f:
    #     json.dump(wrong_list, f, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    seed_everything(931)
    '''加载嵌入模型'''
    model_path='../../LongRAG-main/multilingual-e5-large'
    # index_path = f'./data/corpus/processed/{args.dataset}/vector.index' # Vector index path
    index_path='./data/corpus/processed/wiki/vector.index'
    emb_model = SentenceTransformer(model_path)
    vector = faiss.read_index(index_path)
    question_file_path=f'./data/corpus/raw/{args.dataset}.json'
    # 读取文件
    with open(question_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 提取所有问题的 id
    ids = [item['id'] for item in data]
    # 2. 加载 commonsense_qa 数据集
    if args.dataset=="commonsense_qa":
        ds = load_dataset("./data/commonsense_qa")
        train_data = ds['train']
    # 根据 id 查找问题的选项和正确答案
    result = []
    for item in train_data:
        if item['id'] in ids:
            question=item['question']
            question_id = item['id']
            choices = item['choices']
            correct_answer = item['answerKey']
            
            result.append({
                'id': question_id,
                'question':question,
                'choices': choices,
                'answerKey': correct_answer
            })
    if args.method=="raw":
        print("raw")
        raw_answer(result)
    if args.method=="rag":
        print("rag")
        rag_answer(result)