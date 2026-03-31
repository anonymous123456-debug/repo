import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json
import os

env = os.environ.copy()
env["PYTHONHTTPSVERIFY"] = "0"   # 忽略 SSL 验证


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

def extract_final_option(text: str) -> str:
    marker = "SUCCESS: Local Search Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()

ds = load_ds_mcqa('../dataset/mcqa.json')
ds = ds[:200]

correct=0
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    answer=item["answer"]
    context=item['context']
    question=item['question']
    options=item['choices']
    query_text = f"""
        You are a helpful assistant for a multiple-choice question answering task.

        Please analyze the question based on the context. Then select the best answer from the four choices.

        Important: Output only one of the following: "choice_1", "choice_2", "choice_3", or "choice_4". Do not include any punctuation, explanation, or other text.
        
        Context: {context}
        Question: {question}
        Choices:
        choice_1: {options['choice_1']}
        choice_2: {options['choice_2']}
        choice_3: {options['choice_3']}
        choice_4: {options['choice_4']}
    """  # 你的查询内容
    command = [
        "graphrag", "query",
        "--root", "/share/home/ncu_418000240001/graphrag",
        "--method", "local",
        "--query", query_text
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    # print("======== subprocess debug ========")
    # print(f'[COMMAND]: {" ".join(command)}')
    # print(f'[STDOUT]: {result.stdout}')
    # print(f'[STDERR]: {result.stderr}')
    # print(f'[RETURN CODE]: {result.returncode}')
    # print("===================================")
    response=result.stdout # 查询结果
    print("query:"+query_text)
    print(f'原始输出：{response}\n')
    response=extract_final_option(response)
    print(f'predict:{response}\n')
    if answer.lower() in response.lower():
        print("成功答对一道题！\n")
        correct+=1
    else:
        print(f'wrong!!!! predict: {response}  true: {answer}\n')
    overall_progress.update(1)
end_time=time.time()
print(f'ACC:{correct/200} time:{float(end_time-start_time)}')
overall_progress.close()