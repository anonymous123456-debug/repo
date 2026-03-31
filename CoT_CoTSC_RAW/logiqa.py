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

def remove_trailing_period(s):
    if s.endswith('.'):
        return s[:-1]  # 去除尾部的句号
    else:
        return s  # 不做任何改变
def load_ds(filepath):
    result=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
         result.append({
                'id':d['id'],
                'context':d['context'],
                'options':d['options'],
                'answer': d['correct_option'],
                'question': d['query']
            })
    return result
def load_ds_mcqa(filepath):
    result=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
         result.append({
                'id':d['id'],
                'choices':d['choices'],
                'context':d['context'],
                'answer': d['answer'],
                'question': d['question']
            })
    return result
def get_cot_answer(text):
   # 将文本按行分割
    lines = text.strip().split("\n")
    # 获取最后一行
    last_line = lines[-1].strip()
    return last_line
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
def raw_answer(results):
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        Suppose you are a civil servant, your logical reasoning ability is very strong. Below I will ask you some national civil service exam questions, which are all based on context. Please choose the final answer from the given four options (choices_1,choices_2,choices_3,choices_4) according to the context and the question. The final answer requires only the output options (choices_1,choices_2,choices_3,choices_4) and does not contain any other words.
        
        Context:
        {q['context']}

        Question:
        {q['question']}

        Options:
        choices_1:{q['options']['choices_1']}
        choices_2:{q['options']['choices_2']}
        choices_3:{q['options']['choices_3']}
        choices_4:{q['options']['choices_4']}

        answer:
        """
        # print(f'prompt:{prompt}\n')
        print(f'prompt: {prompt}\n')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response=remove_trailing_period(response)
        print(f'response: {response}\n')
        if q['answer'].lower() in response:
            correct_answer+=1
            correct_list.append(q['id'])
            print('yes\n')
        else:
            wrong_list.append(q['id'])
            print(f'response---->{response}----->answer{q["answer"]}\n')
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
    with open(f'./analysis/raw/loiqa_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/raw/logiqa_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
def cot_answer(results):
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        Suppose you are a civil servant, and your logical reasoning skills are strong. Now I'm going to ask you some questions about the national civil service exam, which are all based on context. According to the given context and the four options, use the thinking chain to reason step by step (the reasoning steps should not exceed 20 steps), and output the final answer, only need to output the options (choices_1, choices_2, choices_3, choices_4).
        
        Context:
        {q['context']}

        Question:
        {q['question']}

        Options:
        choices_1:{q['options']['choices_1']}
        choices_2:{q['options']['choices_2']}
        choices_3:{q['options']['choices_3']}
        choices_4:{q['options']['choices_4']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        Final answer(choices_1 or choices_2 or choices_3 or choices_4):
        """
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'cot是：----{response}\n')
        response=get_cot_answer(response)
        print(f'最后的结论是:{response}\n')
        if q["answer"].lower() in response.lower():
            correct_answer+=1
            print('正确\n')
            correct_list.append(q['id'])
        else:
            wrong_list.append(q['id'])
            print(f'response---->{response}----->true_answer{q["answer"]}\n')
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
    with open(f'./analysis/cot/logiqa_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot/logiqa_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
def cot_sc_answer(results, num_chains=5):
    correct_answer = 0
    total_answer = len(results)
    correct_list = []
    wrong_list = []
    
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time = time.time()
    
    for q in results:
        # 保存所有生成的思维链
        chain_responses = []
        
        # 对每个问题生成多个思维链
        for _ in range(num_chains):
            # 构造 Prompt
            prompt = f"""
        Suppose you are a civil servant, and your logical reasoning skills are strong. Now I'm going to ask you some questions about the national civil service exam, which are all based on context. According to the given context and the four options, use the thinking chain to reason step by step (the reasoning steps should not exceed 20 steps), and output the final answer, only need to output the options (choices_1, choices_2, choices_3, choices_4).
        
        Context:
        {q['context']}

        Question:
        {q['question']}

        Options:
        choices_1:{q['options']['choices_1']}
        choices_2:{q['options']['choices_2']}
        choices_3:{q['options']['choices_3']}
        choices_4:{q['options']['choices_4']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        Final answer(choices_1 or choices_2 or choices_3 or choices_4):
            """
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000, pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f'cot是：---{response}\n')
            response=get_cot_answer(response)
            print(f'最后的结论是:---- {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {'choices_1': 0, 'choices_2': 0,'choices_3':0,'choices_4':0}
        for chain in chain_responses:
            if 'choices_1' in chain.lower():
                option_counts['choices_1'] += 1
            elif 'choices_2' in chain.lower():
                option_counts['choices_2'] += 1
            elif 'choices_3' in chain.lower():
                option_counts['choices_3'] += 1
            elif 'choices_4' in chain.lower():
                option_counts['choices_4'] += 1
            else :
                print('not match')
        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
    
        # 判断最终答案与正确答案是否一致
        if q["answer"].lower() in predicted_answer.lower():
            correct_answer += 1
            correct_list.append(q['id'])
            print('正确\n')
        else:
            wrong_list.append(q['id'])
            print(f'response---->{response}----->true_answer{q["answer"]}\n')
            

        overall_progress.update(1)
            # 将正确和错误回答的问题保存到JSON文件

    end_time = time.time()
    cost = float(end_time - start_time)
    print(f'time: {cost:.2f} s')
    print(f'total_answer: {total_answer}')
    print(f'correct_answer: {correct_answer}')
    print(f'Accuracy: {correct_answer / total_answer}')
    # 关闭总体进度条
    overall_progress.close()
    with open(f'./analysis/cot_sc/logicqa_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot_sc/logicqa_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)



   
      

  
if __name__ == '__main__':
    seed_everything(931)
    ds = load_ds('./data/logiqa/logiqa_2000.json')
    '''
    原始 cot 和 cot-sc tot
    '''
    if args.method=="raw":
        raw_answer(ds)
    if args.method=="cot":
        cot_answer(ds)
    if args.method=="cotsc":
        cot_sc_answer(ds)
    
    