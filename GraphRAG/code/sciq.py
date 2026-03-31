import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json

def extract_final_option(text):
    # match = re.search(r'final_option:\s*([A-Z])', text)
    # return match.group(1) if match else 'NOT MATCH'
    # 尝试提取 `final_option:` 后的大写字母
    match = re.search(r'final_option:\s*([A-Z])', text)
    if match:
        return match.group(1)
    
    # 取最后一行
    lines = text.strip().split("\n")
    if lines:
        last_line = lines[-1]
        match_last = re.search(r':\s*([A-Z])', last_line)
        if match_last:
            return match_last.group(1)
    
    # 如果仍然找不到，返回 'A'
    return 'NOT MATCH'


with open(f'../dataset/sciq.json', encoding='utf-8') as f:
    ds = json.load(f)
    ds=ds[:200]

correct=0
label_mapping = {"choices_1": "A", "choices_2": "B", "choices_3": "C", "choices_4": "D"}
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    quesiton=item['question']
    answer=label_mapping[item['correct_answer']]
    query_text = f"""
    Question:
        {quesiton}
    Options:
        A.{item['choices_1']}
        B.{item['choices_2']}
        C.{item['choices_3']}
        D.{item['choices_4']}
    Please select the best answer according to the question and the above options, and output the uppercase option letters of the answer (A,B,C,D), and output your answer on the last line of your reasoning in the format: final_option:.
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
    response=extract_final_option(response)
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