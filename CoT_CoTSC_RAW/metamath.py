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
def get_response_answer(text):
    match = re.search(r'The answer is:\s*(.*)', text)
    if match:
        answer = match.group(1)
        return answer
    else:
        print('not found')
        return 'none'
def modify_string(string):
    #将第一个单词换成小写
    # 拆分字符串为单词
    words = string.split()
    
    # 如果字符串有单词，修改第一个单词为小写，其他单词不变
    if words:
        words[0] = words[0].lower()
    
    # 重新组合成字符串并返回
    return ' '.join(words)
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
                'answer': d['response'],
                'question': d['query'],
                'id':d['id']
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
    # last_line = remove_trailing_period(last_line).lower()
    # 使用正则表达式匹配最后一行的 "yes" 或 "no"
    # match = re.search(r"(yes|no)$", last_line)

    # # 如果匹配成功，提取结论
    # if match:
    #     conclusion = match.group(1)  # 获取匹配的 "yes" 或 "no"
    #     return conclusion
    # else:
    #     if 'yes' in last_line.lower():
    #         return 'yes'
    #     else: return 'no'
    #     print(f'为啥notfound---{last_line}\n')
    #     print("not found")
    #     return "no"
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
        If you are an expert in the field of mathematics, I will ask you some mathematical questions, please give the answers directly, without giving any other words.

        Question:
        {q['question']}
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
        answer=get_response_answer(q['answer'])
        print(f'answer:{answer}\n')
        if answer.lower() in response.lower():
            correct_answer+=1
            correct_list.append(f'quesiton:->>+{q["question"]}')
            print('yes\n')
        else:
            wrong_list.append(f'quesiton:->>+{q["question"]}')
            print(f'response---->{response}----->answer{answer}\n')
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
    with open(f'./analysis/raw/metamath_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/raw/metamath_wrong_answers.json', 'w', encoding='utf-8') as f:
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
        If you are an expert in the field of mathematics, I will ask you some mathematical questions, please use the chain of thought to think step by step about the solution of the problem, and give the final answer.

        Question:
        {q['question']}

        final answer:
        """
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'cot是:{response}\n')
        response=get_cot_answer(response)
        print(f'response是:{response}\n')
        answer=get_response_answer(q['answer'])
        if answer.lower() in response.lower():
            correct_answer+=1
            print('正确\n')
            correct_list.append(q['id'])
        else:
            wrong_list.append(q['id'])
            print(f'answer---->{response}----->true_answer{answer}\n')
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
    with open(f'./analysis/cot/metamath_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot/metamath_wrong_answers.json', 'w', encoding='utf-8') as f:
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
        If you are an expert in the field of mathematics, I will ask you some mathematical questions. Please think about the solution steps step by step with the chain of thinking, and give the final answer. The output format of the answer is "final answer:".

        Question:
        {q['question']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        final answer:
        """
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.get('attention_mask'), max_new_tokens=10000, pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f'cot是：{response}\n')
            response=get_cot_answer(response)
            print(f'response是: {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {}
        for chain in chain_responses:
             # 使用正则表达式提取 'final answer: ' 后面的部分
            match = re.search(r'final answer:\s*(\S+)', chain.lower())  # \S+ 表示匹配非空白字符
            if match:
                answer = match.group(1)  # 提取匹配的答案部分
                if answer in option_counts:
                    option_counts[answer] += 1
                else:
                    option_counts[answer] = 1
            else:
                print('not match answer\n')
                option_counts['none'] = 1
        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
        true_answer=get_response_answer(q['answer'])
        # 判断最终答案与正确答案是否一致
        if true_answer.lower() in predicted_answer.lower():
            correct_answer += 1
            correct_list.append(q['id'])
            print('正确\n')
        else:
            wrong_list.append(q['id'])
            print(f'predict---->{predicted_answer}----->true_answer{true_answer}\n')
            

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
    with open(f'./analysis/cot_sc/metamath.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot_sc/metamath.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)

              
if __name__ == '__main__':
    seed_everything(931)
    ds = load_ds('./data/metamath/MetaMathQA.json')
    '''
    原始 cot 和 cot-sc tot
    '''
    if args.method=="raw":
        raw_answer(ds)
    if args.method=="cot":
        cot_answer(ds)
    if args.method=="cotsc":
        cot_sc_answer(ds)
 
    