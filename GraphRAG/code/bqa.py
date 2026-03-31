import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json

def load_bqa_ds(filepath):
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

def extract_final_option(text: str) -> str:
    marker = "SUCCESS: Local Search Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()

ds = load_bqa_ds('../dataset/bqa.json')
ds = ds[:200]

correct=0
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    answer=item["answer"]
    context=item['context']
    question=item['question']
    query_text = f"""
            You are a helpful assistant answering binary (yes/no) questions based on the provided information.
            
            Please reason over both the context and then answer the question. Your answer must be strictly one word: "yes" or "no", with no other output.
            
            Context: {context}
            Question: {question}
    """  # 你的查询内容
    command = [
        "graphrag", "query",
        "--root", "/share/home/ncu_418000240001/graphrag",
        "--method", "local",
        "--query", query_text
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print("======== subprocess debug ========")
    print(f'[COMMAND]: {" ".join(command)}')
    print(f'[STDOUT]: {result.stdout}')
    print(f'[STDERR]: {result.stderr}')
    print(f'[RETURN CODE]: {result.returncode}')
    print("===================================")
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