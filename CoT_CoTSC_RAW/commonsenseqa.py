

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
parser.add_argument("--dataset", type=str,help="Name of the dataset")
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
    feature = emb_model.encode([question])
    # 粗粒度查询1个chunks
    distance, match_id = vector.search(feature, 1)
    # 获取chunks
    content = [chunk_data[int(i)] for i in match_id[0]]
    # 返回chunks和对应的id值 用于映射p
    return content, list(match_id[0])
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
def get_cot_answer(text):
     # 按行拆分文本
    if text is None:
        return "none" 
    lines = text.split("\n")
     # 获取最后一行
    last_line = lines[-1].strip()
    return last_line
def remove_trailing_period(s):
    if s.endswith('.'):
        return s[:-1]  # 去除尾部的句号
    else:
        return s  # 不做任何改变
def raw_answer(results):
    global total_prompt_tokens ,total_completion_tokens
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
         # 统计 prompt token 数
        prompt_tokens = len(model_inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # 统计 completion token 数
        completion_tokens = len(generated_ids[0])
        total_completion_tokens += completion_tokens
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
         # 统计 prompt token 数
        prompt_tokens = len(model_inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
         # 统计 completion token 数
        completion_tokens = len(generated_ids[0])
        total_completion_tokens += completion_tokens
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
    with open(f'./analysis/rag/{args.dataset}_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/rag/{args.dataset}_wrong_answers.json', 'w', encoding='utf-8') as f:
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
        content,list=vector_search(q['question'])
        prompt=f'''
        prompt = (
            "Assuming you are a know-it-all, I will now ask you some common sense questions"
            "You will derive the correct answer by reasoning through the provided information. "
             Please deduce the answer step by step according to the thought chain (no more than 20 steps of reasoning).
            "You will be given four possible answer choices: A, B, C, D,and E. Please choose the best answer from these options. "
            
            Question:
            {q['question'] }
            Options:
                A.{q['choices']['text'][0]}
                B.{q['choices']['text'][1]}
                C.{q['choices']['text'][2]}
                D.{q['choices']['text'][3]}
            Now, reason through the question step by step and provide the correct answer.
            The result only needs to output the answer capital option letter (A,B,C,D,E).
            Importantly, the last step of reasoning in the chain of thought outputs only the conclusion, without any words. The Conclusion format is: Conclusion:()
        '''
        print(f'prompt:{prompt}\n')
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
        print(f'cot是：{response}\n')
        response=get_cot_answer(response)
        response=extract_conclusion(response)
        print(f'response是:{response}\n')
        if q["answerKey"] in response:
            correct_answer+=1
            correct_list.append(q['question'])
        else:
            wrong_list.append(q['question'])
            print(f'response:{response}---answer:{q["answerKey"]}\n')
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
def cot_sc_answer(results,num_chains=5):
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
            "Assuming you are a know-it-all, I will now ask you some common sense questions based on the given context. "
            "You will derive the correct answer by reasoning through the provided information. "
             Please deduce the answer step by step according to the thought chain (no more than 20 steps of reasoning).
            "You will be given four possible answer choices: A, B, C, D,and E. Please choose the best answer from these options. "
            "Additionally, I will provide some extra background knowledge to assist in your reasoning. "
            
            Question:
            {q['question'] }
            Options:
                A.{q['choices']['text'][0]}
                B.{q['choices']['text'][1]}
                C.{q['choices']['text'][2]}
                D.{q['choices']['text'][3]}
            Now, reason through the question step by step and provide the correct answer.
            The result only needs to output the answer capital option letter (A,B,C,D,E).
            Importantly, the last step of reasoning in the chain of thought outputs only the conclusion, without any words. The Conclusion format is: Conclusion:()
            
            """
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
             # 统计 prompt token 数
            prompt_tokens = len(model_inputs.input_ids[0])
            total_prompt_tokens += prompt_tokens
            generated_ids = model.generate(model_inputs.input_ids, attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000, pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
             # 统计 completion token 数
            completion_tokens = len(generated_ids[0])
            total_completion_tokens += completion_tokens
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f'cot: {response}\n')
            response=extract_conclusion(response)
            response=remove_trailing_period(response)
            print(f'response: {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {'A': 0, 'B': 0,'C':0,'D':0,'E':0}
        for chain in chain_responses:
            if 'A' in chain:option_counts['A']+=1
            elif 'B' in chain:option_counts['B']+=1
            elif 'C' in chain:option_counts['C']+=1
            elif 'D' in chain:option_counts['D']+=1
            elif 'E' in chain:option_counts['E']+=1
            else:print('not match abcd\n')

        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
    
        # 判断最终答案与正确答案是否一致
        if predicted_answer == q["answerKey"]:
            correct_answer += 1
            print('正确\n')
        else:
            print(f'Prediction ----> {predicted_answer} -----> True answer: {q["answerKey"]}\n')
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


if __name__ == '__main__':
    global total_prompt_tokens ,total_completion_tokens
    total_prompt_tokens = 0        # ← 必须显式初始化
    total_completion_tokens = 0    # ← 必须显式初始化
    seed_everything(931)
    '''加载嵌入模型'''
    model_path='../../LongRAG-main/multilingual-e5-large'
    index_path = f'./data/corpus/processed/commonsense_qa/vector.index' # Vector index path
    emb_model = SentenceTransformer(model_path)
    vector = faiss.read_index(index_path)
    with open(f"./data/corpus/processed/commonsense_qa/chunks.json", "r") as fin:
        chunk_data = json.load(fin)
    # question_file_path=f'./data/corpus/raw/{args.dataset}.json'
    # # 读取文件
    # with open(question_file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # # 提取所有问题的 id
    # ids = [item['id'] for item in data]
    # 2. 加载 commonsense_qa 数据集
    
    ds = load_dataset("./data/commonsense_qa")['train']
    ds= ds.select(range(5))
    # 根据 id 查找问题的选项和正确答案
    result = []
    for item in ds:
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
    if args.method=="cot":
        print("rag")
        cot_answer(result)
    if args.method=="cotsc":
        print("rag")
        cot_sc_answer(result)

    print(f'Total prompt tokens: {total_prompt_tokens}')
    print(f'Total completion tokens: {total_completion_tokens}')
    print(f'Total tokens: {total_prompt_tokens + total_completion_tokens}')