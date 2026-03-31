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
# modelname='../llama3-8b'
# modelname='/share/home/ncu_418000240001/qwen-1.5b'
# modelname='/share/home/ncu_418000240001/smolLM2-1.7b'
modelname='/share/home/ncu_418000240001/phi-3.8b'
# modelname='/share/home/ncu_418000240001/qwen2.5-7b'
# modelname='/share/home/ncu_418000240001/mistral'


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
                'question':d['question'],
                'choices_1':d['choices_1'],
                'choices_2':d['choices_2'],
                'choices_3':d['choices_3'],
                'choices_4':d['choices_4'],
                'answer':d['correct_answer']
            })
    return result
def vector_search(question):
    feature = emb_model.encode([question])
    # 粗粒度查询1个chunks
    distance, match_id = vector.search(feature,10)
    # 获取chunks
    content = [chunk_data[int(i)] for i in match_id[0]]
    # 返回chunks和对应的id值 用于映射p
    return content, list(match_id[0])
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
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        # 构造 Prompt
        prompt = f"""
        If you are an expert in the field of science, I will ask you a number of individual choice questions in the field of science, please choose the best answer from the given four options (choices_1,choices_2,choices_3,choices_4). Please give the answer options directly without typing out any other words.

        Question:
        {q['question']}

        Options:
        choices_1.{q['choices_1']}
        choices_2.{q['choices_2']}
        choices_3.{q['choices_3']}
        choices_4.{q['choices_4']}

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
        if(q['answer'].lower() in response):
            correct_answer+=1
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    print(f'time:{cost:.2f} s')
    print(f'total_answer:{total_answer}')
    print(f'correct_answer:{correct_answer}')
    print(f'准确率:{correct_answer/total_answer}')
    # 计算所有指标的平均分数
    # 关闭总体进度条
    overall_progress.close()
    
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
        If you're an expert in science, I'm going to ask you some multiple-choice questions in science, and your task is to deduce the answer to the question step by step (no more than 20 steps) along the chain of thought. And choose the best answer from the four options given (choices_1,choices_2,choices_3,choices_4).
        In addition, I will give you a sample answer for you to learn from.
        
        Question:
        {q['question']}

        Options:
        choices_1.{q['choices_1']}
        choices_2.{q['choices_2']}
        choices_3.{q['choices_3']}
        choices_4.{q['choices_4']}

        sample:
        thought process:
        1. The question is asking about the flow of charge that results from potential differences from various voltage sources.
        2. Potential differences refer to the difference in electric potential between two points, which is essentially what we call voltage.
        3. Voltage is a measure of the potential difference that drives electric charge from one point to another.
        4. When there is a potential difference, or voltage, across a conductor, it creates an electric field.
        5. The electric field is a region around the conductor where electric forces can act on charged particles.
        6. The presence of an electric field causes charged particles, such as electrons, to move.
        7. This movement of charged particles is what we call electric current.
        8. Therefore, the flow of charge that results from potential differences from various voltage sources is electric current.
        So, based on this reasoning process, the answer is:choices_1.current
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
        response=get_cot_answer(response).lower()
        print(f'结果是：{response}\n')
        if(q['answer'].lower() in response):
            correct_answer+=1
            print('正确\n')
        else: print(f'prediction:--{response}\nground:--{q["answer"]}\n')
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    print(f'time:{cost:.2f} s')
    print(f'total_answer:{total_answer}')
    print(f'correct:{correct_answer}\n')
    print(f'correct:{correct_answer/total_answer}\n')
    # 关闭总体进度条
    overall_progress.close()
       
    
def rag_answer(results):
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    prediction=[]
    ground_true=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        content,list=vector_search(q['question'])
        merged_content = "\n".join([f"evidence {idx + 1}：{chunk['text']}" for idx, chunk in enumerate(content)])
        print(q['question'])
        print('\n')
        print(merged_content)
        print('\n')

        # 构造 Prompt
        prompt = f"""
        If you are an expert in the field of science, I will ask you a number of individual choice questions in the field of science, please choose the best answer from the given four options (choices_1,choices_2,choices_3,choices_4). Please give the answer options directly without typing out any other words.
        Please use the following external knowledge to answer the question.
        
        external knowledge:
        {merged_content}

        Question:
        {q['question']}

        Options:
        choices_1.{q['choices_1']}
        choices_2.{q['choices_2']}
        choices_3.{q['choices_3']}
        choices_4.{q['choices_4']}

        Please give the answer options directly without typing out any other words.
        """
        # print(f'prompt:{prompt}\n')
        print(f'prompt: {prompt}\n')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response=remove_trailing_period(response)
        if(q['answer'].lower() in response.lower()):
            correct_answer+=1
            print('正确\n')
        print(f'prediction:--{response}\nground:--{q["answer"]}\n')
        overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    print(f'time:{cost:.2f} s')
    print(f'correct_answer:{correct_answer}')
    print(f'total_answer:{total_answer}')
    overall_progress.close()

def rag2_answer(results):
    correct_answer=0
    total_answer=len(results)
    correct_list=[]
    wrong_list=[]
    prediction=[]
    ground_true=[]
    overall_progress = tqdm(total=len(results), desc="Overall Progress", unit="task")
    start_time=time.time()
    for q in results:
        content,list=vector_search(q['question'])
        # 构造 Prompt
        prompt = f"""
        Please answer the question in context and use the following external knowledge.
        
        Context:
         {q['context']}

        external knowledge:
        {content[0]['text']}

        Question:
        {q['question']}

        """
        # print(f'prompt:{prompt}\n')
        print(f'prompt: {prompt}\n')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
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
    with open('./analysis/evaluate/squad_rag2.json', 'w') as f:
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
        If you're an expert in science, I'm going to ask you some multiple-choice questions in science, and your task is to deduce the answer to the question step by step (no more than 20 steps) along the chain of thought. And choose the best answer from the four options given (choices_1,choices_2,choices_3,choices_4).
        In addition, I will give you a sample answer for you to learn from.
        
        Question:
        {q['question']}

        Options:
        choices_1.{q['choices_1']}
        choices_2.{q['choices_2']}
        choices_3.{q['choices_3']}
        choices_4.{q['choices_4']}

        sample:
        thought process:
        1. The question is asking about the flow of charge that results from potential differences from various voltage sources.
        2. Potential differences refer to the difference in electric potential between two points, which is essentially what we call voltage.
        3. Voltage is a measure of the potential difference that drives electric charge from one point to another.
        4. When there is a potential difference, or voltage, across a conductor, it creates an electric field.
        5. The electric field is a region around the conductor where electric forces can act on charged particles.
        6. The presence of an electric field causes charged particles, such as electrons, to move.
        7. This movement of charged particles is what we call electric current.
        8. Therefore, the flow of charge that results from potential differences from various voltage sources is electric current.
        So, based on this reasoning process, the answer is:choices_1.current
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
            print(f'cot是：{response}\n')
            response=get_cot_answer(response).lower()
            print(f'结果是: {response}\n')
            chain_responses.append(response)
        
        # 根据生成的多条思维链，选择最一致的答案
        # 统计每个选项的出现次数
        option_counts = {'choices_1':0,'choices_2':0,'choices_3':0,'choices_4':0}
        for chain in chain_responses:
             # 使用正则表达式提取 'final answer: ' 后面的部分
            if('choices_1' in chain):option_counts['choices_1']+=1
            elif('choices_2' in chain):option_counts['choices_2']+=1
            elif('choices_3' in chain):option_counts['choices_3']+=1
            elif('choices_4' in chain):option_counts['choices_4']+=1
            else:
                print('not match answer\n')
        # 选择出现次数最多的选项作为最终答案
        predicted_answer = max(option_counts, key=option_counts.get)
        true_answer=q['answer']
        # 判断最终答案与正确答案是否一致
        if true_answer in predicted_answer:
            correct_answer += 1
            print('正确\n')
        else:
            print(f'predict---->{predicted_answer}----->true_answer{true_answer}\n')
            
        overall_progress.update(1)
            # 将正确和错误回答的问题保存到JSON文件

    end_time = time.time()
    cost = float(end_time - start_time)
    print(f'time: {cost:.2f} s')
    print(f'total_answer: {total_answer}\n')
    print(f'correct_answer: {correct_answer}\n')
    print(f'Accuracy: {correct_answer / total_answer}')
    # 关闭总体进度条
    overall_progress.close()


   
      

  
if __name__ == '__main__':
    global total_prompt_tokens ,total_completion_tokens
    total_prompt_tokens=0
    total_completion_tokens=0
    seed_everything(931)
    ds = load_ds('./data/sciq/sciq_2000.json')[:5]
    model_path='../../LongRAG-main/multilingual-e5-large'
    index_path = f'./data/corpus/processed/sciq/vector.index' # Vector index path
    emb_model = SentenceTransformer(model_path)
    vector = faiss.read_index(index_path)
    with open(f"./data/corpus/processed/sciq/chunks.json", "r") as fin:
        chunk_data = json.load(fin)
    '''
    原始 cot 和 cot-sc tot
    '''
    if args.method=="raw":
        raw_answer(ds)
    if args.method=="cot":
        cot_answer(ds)
    if args.method=="rag":
        rag_answer(ds)
    if args.method=="rag2":
        rag2_answer(ds)
    if args.method=="cotsc":
        cot_sc_answer(ds)
    print(f'total_prompt_tokens:{total_prompt_tokens}')
    print(f'total_completion_tokens:{total_completion_tokens}')
    print(f'total_tokens:{total_prompt_tokens+total_completion_tokens}')
    
    