import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json

def extract_final_choice(text):
    # 规则1：匹配 Final_??? 或 Final ??? 后的 A-E（支持英文或中文冒号）
    match = re.search(r'\b[Ff]inal[_ ]\w+[：:]\s*([A-E])\b', text)
    if match:
        return match.group(1)

    # 规则3：匹配最后一行的 A-E
    lines = text.strip().splitlines()
    if lines:
        last_line = lines[-1]
        match_last = re.search(r'\b([A-E])\b', last_line)
        if match_last:
            return match_last.group(1)

    # 规则2：匹配 "answer is X" 结构
    match_answer = re.search(r'answer is\s+([A-E])\b', text, re.IGNORECASE)
    if match_answer:
        return match_answer.group(1).upper()

    # 都没有匹配到
    return "none"

ds = load_dataset('csv', data_files='../dataset/cosmosqa/train.csv')['train']
ds = ds.select(range(0,200))

correct=0
label_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    quesiton=item['question']
    answer=label_mapping[item["label"]]
    context=item['context']
    query_text = f"""
    Question:
        {quesiton}
    Options:
        A.{item['answer0']}
        B.{item['answer1']}
        C.{item['answer2']}
        D.{item['answer3']}
    Context:
        {context}
    Please select the best answer according to the question and the above options and the Context, and output the uppercase option letters of the answer (A,B,C,D), and output your answer on the last line of your reasoning in the format: final_option:.
    """  # 你的查询内容
    command = [
        "graphrag", "query",
        "--root", "/share/home/ncu_418000240001/graphrag",
        "--method", "local",
        "--query", query_text
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    response=result.stdout # 查询结果
    print(f'原始输出：{response}\n')
    response=extract_final_choice(response)
    print(f'predict: {response}  true: {answer}\n')
    if answer in response:
        print("成功答对一道题！\n")
        correct+=1
    else:
        print("错误！！！")
    overall_progress.update(1)
end_time=time.time()
print(f'ACC:{correct/200} time:{float(end_time-start_time)}')
overall_progress.close()