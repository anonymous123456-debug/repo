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
import re
from metric import compute_average_metrics,F1_scorer
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,GPT2Tokenizer, GPT2Model


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str,help="Name of the dataset")
args = parser.parse_args()

'''
加载模型
'''
modelname='../llama3-8b'
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')


def load_ds(filepath):
    result=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    for d in ds:
         result.append({
                'id':d['id'],
                'context':d['context'],
                'answer': d['answers']['text'][0],
                'question': d['question']
            })
    return result

def get_score(text):
    # 使用正则表达式提取数字
    score = re.search(r'\d+', text)
    if score:
        return int(score.group())
    else:
        print("not found score")
        return 0
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
def squad_evaluate(results):
    if args.method=="squad_raw":
        with open(f"./analysis/evaluate/squad_raw.json", "r") as fin:
            pair = json.load(fin)
    if args.method=="squad_cot":
        with open(f"./analysis/evaluate/squad_cot.json", "r") as fin:
            pair = json.load(fin)
    if args.method=="squad_rag":
        with open(f"./analysis/evaluate/squad_rag.json", "r") as fin:
            pair = json.load(fin)
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    prediction=[]
    ground_true=[]
    score=0
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q,p in zip(results,pair):
        # 构造 Prompt
        prompt = f"""
        I'll give you some predicted and standard answers to questions that are context-based. Based on the question and context, evaluate the semantic relevance score between the predicted and standard answers on a scale of 0-100. You only need to output the final score, you don't need to output any extra characters.
        
        Context:
        {q['context']}

        Question:
        {q['question']}

        Predicted Answer:
        {p['prediction']}

        Standard Answer:
        {p['ground']}
        """
        # print(f'prompt:{prompt}\n')
        print(f'prompt: {prompt}\n')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        score+=get_score(response)
        print(f'now score:{score}\n')
        overall_progress.update(1)
   
    # 关闭总体进度条
    overall_progress.close()
    print(f'总分：{score}')
    print(f'均分：{score/2000}')
    

    

    
    


   
      

  
if __name__ == '__main__':
    seed_everything(931)
    ds = load_ds('./data/squad/squad.json')
    squad_evaluate(ds)
    
    
    