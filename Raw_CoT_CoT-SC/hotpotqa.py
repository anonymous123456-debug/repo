import os
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
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str,help="Name of the dataset")
parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"), help="Local model path or Hugging Face model id")
args = parser.parse_args()

'''
加载模型
'''
modelname = args.model_path


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
        ds = json.load(f)
    for d in ds:
         result.append({
                'id':d['question_id'],
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
    global total_prompt_tokens ,total_completion_tokens
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    prediction=[]
    ground_true=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        Please answer the question in context. Just write a short answer, not a long one.
        
        Context:
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
        response=remove_trailing_period(response).lower()
        print(f'prediction:--{response}\nground:--{q["answer"]}\n')
        prediction.append(response.lower())
        ground_true.append(q['answer'].lower())
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    data = [{"prediction": pred, "ground": truth} for pred, truth in zip(prediction, ground_true)]

    # 将数据写入 JSON 文件
    with open('./analysis/evaluate/hotpotqa_raw.json', 'w') as f:
        json.dump(data, f, indent=4)
    print(f'待评估答案已写入analysis')
    print(f'time:{cost:.2f} s')
    print(f'total_answer:{total_answer}')
    # 计算所有指标的平均分数
    average_metrics = compute_average_metrics(prediction, ground_true)
    xx=F1_scorer(prediction,ground_true)
    print(f'基于字符的f1{xx}')
    print(f'em_avg:{average_metrics["exact_match"]}')
    print(f'f1_avg:{average_metrics["f1"]}')
    print(f'bleu_avg:{average_metrics["bleu"]}')
    print(f'rouge_avg:{average_metrics["rouge"]}')
    # 关闭总体进度条
    overall_progress.close()
    
def cot_answer(results):
    global total_prompt_tokens ,total_completion_tokens
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    prediction=[]
    ground_true=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        Your task is to answer the question directly, given the context, and use the chain of thought to reason the answer step by step (no more than 20 steps in the chain of thought reasoning), and give the final answer, only need to output a short answer, no longer a long piece.
        
        Context:
        {q['context']}

        Question:
        {q['question']}

        Thought chain reasoning process:
        1.
        2.
        3.
        ...
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
        print(f'cot是：{response}\n')
        response=get_cot_answer(response)
        print(f'结果是：{response}\n')
        response=remove_trailing_period(response).lower()
        print(f'prediction:--{response}\nground:--{q["answer"]}\n')
        prediction.append(response.lower())
        ground_true.append(q['answer'].lower())
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    data = [{"prediction": pred, "ground": truth} for pred, truth in zip(prediction, ground_true)]

    print(f'time:{cost:.2f} s')
    print(f'total_answer:{total_answer}')
    # 计算所有指标的平均分数
    average_metrics = compute_average_metrics(prediction, ground_true)
    xx=F1_scorer(prediction,ground_true)
    print(f'基于字符的f1{xx}')
    print(f'em_avg:{average_metrics["exact_match"]}')
    print(f'f1_avg:{average_metrics["f1"]}')
    print(f'bleu_avg:{average_metrics["bleu"]}')
    print(f'rouge_avg:{average_metrics["rouge"]}')
    # 关闭总体进度条
    overall_progress.close()

def cot_sc_answer(results, num_samples=5, temperature=0.7):
    global total_prompt_tokens ,total_completion_tokens
    total_answer = len(results)
    prediction = []
    ground_true = []
    overall_progress = tqdm(total=total_answer, desc="CoT-SC Progress", unit="task")
    start_time = time.time()

    for q in results:
        answers = []
        for i in range(num_samples):
            prompt = f"""
            Your task is to answer the question directly, given the context, and use the chain of thought to reason the answer step by step (no more than 20 steps in the chain of thought reasoning), and give the final answer, only need to output a short answer, no longer a long piece.
            
            Context:
            {q['context']}

            Question:
            {q['question']}

            Thought chain reasoning process:
            1.
            2.
            3.
            ...
            """

            print(f'[Sample {i+1}] Prompt:\n{prompt}\n')
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
             # 统计 prompt token 数
            prompt_tokens = len(model_inputs.input_ids[0])
            total_prompt_tokens += prompt_tokens
            # 使用采样生成不同推理链
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.get('attention_mask'),
                do_sample=True,
                temperature=temperature,
                max_new_tokens=400,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            # 统计 completion token 数
            completion_tokens = len(generated_ids[0])
            total_completion_tokens += completion_tokens
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f'[Sample {i+1}] Raw CoT:\n{response}\n')

            # 提取最终答案
            answer = get_cot_answer(response)
            answer = remove_trailing_period(answer).lower()
            answers.append(answer)

        # 多数投票选出最终答案
        final_answer = Counter(answers).most_common(1)[0][0]
        print(f'[Final Answer by Voting] {final_answer}\nGround Truth: {q["answer"]}\n')

        prediction.append(final_answer.lower())
        ground_true.append(q['answer'].lower())
        overall_progress.update(1)

    end_time = time.time()
    cost = float(end_time - start_time)

    print(f'Time Cost: {cost:.2f}s')
    print(f'Total Examples: {total_answer}')

    average_metrics = compute_average_metrics(prediction, ground_true)
    f1 = F1_scorer(prediction, ground_true)

    print(f'Character-level F1: {f1}')
    print(f'Exact Match (EM): {average_metrics["exact_match"]}')
    print(f'F1: {average_metrics["f1"]}')
    print(f'BLEU: {average_metrics["bleu"]}')
    print(f'ROUGE: {average_metrics["rouge"]}')
    overall_progress.close()

    return prediction, ground_true    
       


  


   
      

  
if __name__ == '__main__':
    global total_prompt_tokens ,total_completion_tokens
    total_completion_tokens=0
    total_prompt_tokens=0
    seed_everything(931)
    ds = load_ds('./data/2qa/hotpotqa.json')[:200]
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
    
    
