from datasets import load_dataset
import logging
import random
import faiss
import os
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,GPT2Tokenizer, GPT2Model
import numpy as np
from datetime import datetime
import backoff
import argparse
import yaml
import torch
logger = logging.getLogger()
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import json

'''
生成commonsense_qa的rag语料文档
'''

ds = load_dataset("./data/commonsense_qa")
data = ds['train']
questions = [sample['question'] for sample in data]
ids=[sample['id'] for sample in data]
# 随机选择2000个问题的索引
random_indices = random.sample(range(len(questions)), 2000)
# 根据选中的索引选择对应的questions和ids
selected_questions = [questions[i] for i in random_indices]
selected_ids = [ids[i] for i in random_indices]
texts=[]

'''
加载模型
'''
modelname='../llama3-8b'
'''
产生背景知识 生成rag语料
'''
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
overall_progress = tqdm(total=len(selected_questions), desc="Overall Progress", unit="task")
for q in selected_questions:
    #commonsense_qa prompt
    # prompt="Please provide background information on the following question and analyze the answer to that question from multiple perspectives. The result is presented in a single paragraph without blank lines.\n Q:"+q
    #cosmosqa_prompt
    prompt="I'm going to give you a passage and ask you a question about that passage. Make a comprehensive analysis of the text, connect it to the question, and think about what additional information you should retrieve to help you answer the question. Do not answer in separate lines, but in consecutive words.\nContext:"+context+"\nQuestion:"+q
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    texts.append(response)
    print('---response----')
    print(response)
    overall_progress.update(1)

# 关闭总体进度条
overall_progress.close()
json_data = [{"id": id, "question": question,"text":text} for id, question,text in zip(selected_ids, selected_questions,texts)]
with open('output.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

