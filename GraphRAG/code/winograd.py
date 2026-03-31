import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json

def extract_final_option(text: str) -> str:
    marker = "SUCCESS: Local Search Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()

ds = load_dataset("../dataset/winograd")['test']
ds = ds.select(range(0,200))

correct=0
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    answer=item["rendered_output"]
    query_text = f"""
            You are an expert in natural language understanding. Use the retrieved knowledge and the passage to determine what the pronoun refers to.

            Passage:  
            "{item['text']}"

            Pronoun to resolve:  
            "{item['pronoun']}"

            Candidate referents:  
            - "{item['options'][0]}"  
            - "{item['options'][1]}"

            Question:  
            In the passage above, does the pronoun "{item['pronoun']}" refer to "{item['options'][0]}" or "{item['options'][1]}"?

            Answer (Output only one of the two options exactly as written above, with no explanation): 
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