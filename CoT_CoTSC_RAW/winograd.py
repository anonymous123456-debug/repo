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
# modelname='../llama3-8b'
# modelname='/share/home/ncu_418000240001/qwen-1.5b'
# modelname='/share/home/ncu_418000240001/smolLM2-1.7b'
modelname='/share/home/ncu_418000240001/phi-3.8b'
# modelname='/share/home/ncu_418000240001/qwen2.5-7b'
# modelname='/share/home/ncu_418000240001/mistral'


tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
def extract_last_refers_to(text):
    # 按行拆分文本
    if text is None:
        return "none" 
    lines = text.split("\n")
    # 遍历每一行，寻找包含"refers to"的行
    for line in reversed(lines):  # 从最后一行开始遍历
        match = re.search(r'ref(er|ers) to\s+"([^"]+)"', line)
        if match:
            return match.group(2).strip()  # 返回引号中的内容，即第二个捕获组

    return text  # 如果没有找到符合条件的行，返回None

def extract_refers_to(text):
    if text is None:
        return "none" 
    # 使用正则表达式匹配 "refer to" 或 "refers to" 后的引号中的内容
    match = re.search(r'ref(er|ers) to\s+"([^"]+)"', text)
    
    if match:
        # 返回引号中的内容
        return match.group(2).strip()
    else:
        return text
def get_cot_answer(text):
     # 按行拆分文本
    if text is None:
        return "none" 
    lines = text.split("\n")
     # 获取最后一行
    last_line = lines[-1].strip()
    return last_line
def extract_conclusion(text):
    # 使用正则表达式提取“Conclusion: ”后面的内容，并去掉括号和多余空格
    match = re.search(r'Conclusion:\s*(\((.*?)\)|([^()]*))', text)
    if match:
        # 如果有括号，提取括号内的内容；如果没有括号，直接提取
        conclusion = match.group(2) if match.group(2) else match.group(3)
        if conclusion is None:
            return extract_last_refers_to(text)
        conclusion = extract_refers_to(conclusion)
        return conclusion.strip()
    else:
        text=extract_last_refers_to(text)
        return text
def remove_trailing_period(s):
    if s.endswith('.'):
        return s[:-1]  # 去除尾部的句号
    else:
        return s  # 不做任何改变
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
def raw_answer(results):
    global total_prompt_tokens ,total_completion_tokens
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        You are an expert in natural language understanding. Your task is to resolve pronoun ambiguity in the given passage.

        Passage: "{q['text']}"

        Question: In the passage above, does the pronoun "{q['pronoun']}" refer to "{q['options'][0]}" or "{q['options'][1]}"? 

        Please consider the logical meaning of the sentence, grammatical structure, and common sense reasoning. Provide only the correct answer without any additional explanation.

        Answer:
        """
        # print(f'prompt:{prompt}\n')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
         # 统计 prompt token 数
        prompt_tokens = len(model_inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # 统计 completion token 数
        completion_tokens = len(generated_ids[0])
        total_completion_tokens += completion_tokens
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response=remove_trailing_period(response)
        print(f'response:{response}\n')
        if response.lower()==q["rendered_output"].lower():
            correct_answer+=1
            correct_list.append(q['text'])
        else:
            wrong_list.append(q['text'])
            print(f'response---->{response}----->answer{q["rendered_output"]}\n')
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
    with open(f'./analysis/raw/winograd_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/raw/winograd_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)

def cot_answer(results):
    global total_prompt_tokens ,total_completion_tokens
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        You are an expert in language analysis and natural language understanding. Your task is to resolve the ambiguity of pronouns in a given paragraph through step-by-step reasoning(with no more than 20 reasoning steps).
        Passage:  
        "{q['text']}"

        Question:  
        In the passage above, does the pronoun "{q['pronoun']}" refer to "{q['options'][0]}" or "{q['options'][1]}"?

        Step-by-step reasoning:  
        ...

        Conclusion(Output only {q['options'][0]}" or "{q['options'][1]}, no other description is required)：
        """
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
         # 统计 prompt token 数
        prompt_tokens = len(model_inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # 统计 completion token 数
        completion_tokens = len(generated_ids[0])
        total_completion_tokens += completion_tokens
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response=get_cot_answer(response)
        print(f'response是:{response}\n')
        if q["rendered_output"].lower() in response.lower():
            print("yes\n")
            correct_answer+=1
            correct_list.append(q['text'])
        else:
            wrong_list.append(q['text'])
            print(f"wrong --{response}--{q['rendered_output']}\n")
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
   
 
    
    with open(f'./analysis/cot/winograd_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot/winograd_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)

def cot_sc_answer(results, num_chains=5):
    global total_prompt_tokens ,total_completion_tokens
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
            You are an expert in linguistic analysis and natural language understanding. Your task is to resolve pronoun ambiguity in a given passage by reasoning step by step.
            ### Passage:  
            "{q['text']}"

            ### Question:  
            In the passage above, does the pronoun "{q['pronoun']}" refer to "{q['options'][0]}" or "{q['options'][1]}"?

            Now that I have given the paragraphs to analyze and the corresponding questions, please generate the corresponding thought chain steps.
            Importantly,Only output the Conclusion without any words.The Conclusion format is: Conclusion:()
            """
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            # 统计 prompt token 数
            prompt_tokens = len(model_inputs.input_ids[0])
            total_prompt_tokens += prompt_tokens
            generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            # 统计 completion token 数
            completion_tokens = len(generated_ids[0])
            total_completion_tokens += completion_tokens
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response=extract_conclusion(response)
            response=remove_trailing_period(response)
            print(f'response: {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {q['options'][0]: 0, q['options'][1]: 0}
        for chain in chain_responses:
            if chain.lower() == q['options'][0].lower():
                option_counts[q['options'][0]] += 1
            elif chain.lower() == q['options'][1].lower():
                option_counts[q['options'][1]] += 1
        
        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
    
        # 判断最终答案与正确答案是否一致
        if predicted_answer.lower() == q["rendered_output"].lower():
            correct_answer += 1
            correct_list.append(q['text'])
            print('yes\n')
        else:
            wrong_list.append(q['text'])
            print(f'Prediction ----> {predicted_answer} -----> True answer: {q["rendered_output"]}\n')

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
    with open(f'./analysis/cot_sc/winograd_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot_sc/winograd_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)






if __name__ == '__main__':
    seed_everything(931)
    global total_prompt_tokens ,total_completion_tokens
    total_completion_tokens=0
    total_prompt_tokens=0
    ds = load_dataset("./data/winograd")['test']
    # 取前 100 个样本
    ds= ds.select(range(5))
    '''
    原始 cot 和 cot-sc tot
    '''
    if args.method=="raw":
        raw_answer(ds)
    if args.method=="cot":
        cot_answer(ds)
    if args.method=="cotsc":
        cot_sc_answer(ds)
    print(f'total prompt tokens:{total_prompt_tokens}')
    print(f'total completion tokens:{total_completion_tokens}')
    print(f'total tokens:{total_prompt_tokens+total_completion_tokens}')
    
   
