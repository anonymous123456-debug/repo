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
model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
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
                'context':d['context'],
                'answer': d['answer'],
                'question': d['question']
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
        Determine the answer to the question based on the following premises. Please provide a clear "yes" or "no" without giving additional explanations or words.

        Prerequisites:
        {q['context']}

        Question:
        {q['question']}
        """
        # print(f'prompt:{prompt}\n')
        print(f'prompt: {prompt}\n')
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
        print(f'response: {response}\n')
        if response.lower()==q["answer"].lower():
            correct_answer+=1
            correct_list.append(q['context']+" quesiton:->>"+q['question'])
            print('yes\n')
        else:
            wrong_list.append(q['context']+" quesiton:->>"+q['question'])
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
    with open(f'./analysis/raw/logicbench_bqa_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/raw/logicbench_bqa_wrong_answers.json', 'w', encoding='utf-8') as f:
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
        Please reason step by step according to the following premises, and clearly show each thinking step in the reasoning process(with no more than 20 reasoning steps). Finally, give a clear "yes" or "no" as the final answer according to the inference result. Don't give any additional words or explanations.
        
        Prerequisites:
        {q['context']}

        Question:
        {q['question']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        Final conclusion (only "yes" or "no" is given) :
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
        print('-------reponse结束-------')
        if q["answer"].lower() in response.lower():
            correct_answer+=1
            print('正确\n')
            correct_list.append(q['context']+" quesiton:->>"+q['question'])
        else:
            wrong_list.append(q['context']+" quesiton:->>"+q['question'])
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
    with open(f'./analysis/cot/logicbench_bqa_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot/logicbench_bqa_wrong_answers.json', 'w', encoding='utf-8') as f:
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
        Please reason step by step according to the following premises, and clearly show each thinking step in the reasoning process(with no more than 20 reasoning steps). Finally, give a clear "yes" or "no" as the final answer according to the inference result. Don't give any additional words or explanations.
        
        Prerequisites:
        {q['context']}

        Question:
        {q['question']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        Final conclusion (only "yes" or "no" is given) :
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
            print(f'response: {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {'yes': 0, 'no': 0}
        for chain in chain_responses:
            if 'yes' in chain.lower():
                option_counts['yes'] += 1
            elif 'no' in chain.lower():
                option_counts['no'] += 1
            else :
                print('doubushi')
        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
    
        # 判断最终答案与正确答案是否一致
        if predicted_answer.lower() == q["answer"].lower():
            correct_answer += 1
            correct_list.append(q['context']+" quesiton:->>"+q['question'])
            print('正确\n')
        else:
            wrong_list.append(q['context']+" quesiton:->>"+q['question'])
            print(f'response---->{response}----->answer{q["answer"]}\n')
            

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
    with open(f'./analysis/cot_sc/logicbench_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot_sc/logicbench_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
def raw_answer_mcqa(results):
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
        Determine the answer to the question based on the following premise, I will give the premise, question, and answer options choice_1, choice_2, choice_3, choice_4. Please give the final answer options, no additional explanation or text.

        Prerequisites:
        {q['context']}

        Question:
        {q['question']}

        Options:
        choice_1:{q['choices']['choice_1']}
        choice_2:{q['choices']['choice_2']}
        choice_3:{q['choices']['choice_3']}
        choice_4:{q['choices']['choice_4']}
        """
        # print(f'prompt:{prompt}\n')
        print(f'prompt: {prompt}\n')
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
        print(f'response: {response}\n')
        if q["answer"].lower() in response.lower():
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
    with open(f'./analysis/raw/logicbench_mcqa_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/raw/logicbench_mcqa_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
def cot_answer_mcqa(results):
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
        Please reason step by step according to the following premises, and clearly show each thinking step in the reasoning process(with no more than 10 reasoning steps). I will give the premise, question, and answer options choice_1, choice_2, choice_3, choice_4, please give the final answer options without any words.
        
        Prerequisites:
        {q['context']}

        Question:
        {q['question']}

        Options:
        choice_1:{q['choices']['choice_1']}
        choice_2:{q['choices']['choice_2']}
        choice_3:{q['choices']['choice_3']}
        choice_4:{q['choices']['choice_4']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        Final conclusion (only choice_1,choice_2,choice_3,choice_4) :
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
        print(f'cot: {response}\n')
        response=get_cot_answer(response)
        print(f'response是:{response}\n')
        if q["answer"].lower() in response.lower():
            correct_answer+=1
            print('正确\n')
            correct_list.append(q['id'])
        else:
            wrong_list.append(q['id'])
            print(f'response---->{response.lower()}----->answer{q["answer"]}\n')
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
    with open(f'./analysis/cot/logicbench_mcqa_correct_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot/logicbench_mcqa_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
def cot_sc_answer_mcqa(results,num_chains=5):
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
        Please reason step by step according to the following premises, and clearly show each thinking step in the reasoning process(with no more than 20 reasoning steps). I will give you four answer options, and please choose the best answer from options choice_1, choice_2, choice_3, choice_4 without any words.
        
        Prerequisites:
        {q['context']}

        Question:
        {q['question']}

        Options:
        choice_1:{q['choices']['choice_1']}
        choice_2:{q['choices']['choice_2']}
        choice_3:{q['choices']['choice_3']}
        choice_4:{q['choices']['choice_4']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...

        Final conclusion (only choice_1,choice_2,choice_3,choice_4) :
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
            print(f'response是: {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {'choice_1': 0, 'choice_2': 0,'choice_3':0,'choice_4':0}
        for chain in chain_responses:
            if 'choice_1' in chain.lower():
                option_counts['choice_1'] += 1
            elif 'choice_2' in chain.lower():
                option_counts['choice_2'] += 1
            elif 'choice_3' in chain.lower():
                option_counts['choice_3'] += 1
            elif 'choice_4' in chain.lower():
                option_counts['choice_4'] += 1
            else :
                print('doubushi')
        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
    
        # 判断最终答案与正确答案是否一致
        if predicted_answer.lower() == q["answer"].lower():
            correct_answer += 1
            correct_list.append(q['id'])
            print('正确\n')
        else:
            wrong_list.append(q['id'])
            print(f'predict---->{predicted_answer}----->answer{q["answer"]}\n')
            

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
    with open(f'./analysis/cot_sc/logicbench_answers.json', 'w', encoding='utf-8') as f:
        json.dump(correct_list, f, ensure_ascii=False, indent=4)

    with open(f'./analysis/cot_sc/logicbench_wrong_answers.json', 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    global total_prompt_tokens ,total_completion_tokens
    total_completion_tokens=0
    total_prompt_tokens=0
    seed_everything(931)
    ds = load_ds('./data/logicbench/logic_bqa.json')
    ds= ds[:5]
    '''
    原始 cot 和 cot-sc tot
    '''
    if args.method=="raw":
        raw_answer(ds)
    if args.method=="cot":
        cot_answer(ds)
    if args.method=="cotsc":
        cot_sc_answer(ds)
    if args.method=="raw_mcqa":
        ds = load_ds_mcqa('./data/logicbench/logicbench_mcqa.json')
        ds= ds[:5]
        raw_answer_mcqa(ds)
    if args.method=="cot_mcqa":
        ds = load_ds_mcqa('./data/logicbench/logicbench_mcqa.json')
        ds= ds[:5]
        cot_answer_mcqa(ds)
    if args.method=="cotsc_mcqa":
        ds = load_ds_mcqa('./data/logicbench/logicbench_mcqa.json')
        ds= ds[:5]
        cot_sc_answer_mcqa(ds)
    
    print(f'Total prompt tokens: {total_prompt_tokens}')
    print(f'Total completion tokens: {total_completion_tokens}')
    print(f'Total tokens: {total_prompt_tokens + total_completion_tokens}')