import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json
from metric import F1_scorer

def extract_answer(text: str):
    # 尝试提取 final_answer: 后面的内容
    match = re.search(r"final_answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 如果没有 final_answer，查找 Local Search Response 后第一行非空行
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Local Search Response:" in line:
            # 从下一行开始找第一个非空行
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line:
                    return next_line
            break
    
    return None

with open(f'../dataset/2wikimultihopqa.json', encoding='utf-8') as f:
    ds = json.load(f)
    ds=ds[:200]

f1=0.
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    quesiton=item['question']
    answer=item['answer']
    query_text = f"""
    You are a helpful assistant.

    Read the question and give a very short answer.
    Do NOT explain. Do NOT repeat the question. Do NOT use examples from above. 
    Only answer the specific question shown.

    Answer with 1-2 words, no punctuation.

    Question: {quesiton}
    Answer:"""


    command = [
        "graphrag", "query",
        "--root", "/share/home/ncu_418000240001/graphrag",
        "--method", "local",
        "--query", query_text
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    response=result.stdout # 查询结果
    print(f'原始输出：{response}\n')
    response=extract_answer(response)
    print(f'提取输出：{response}\n')
    print(f'answer:{answer}')
    if response!='not match':
        ff=F1_scorer([response],[answer])
        f1+=ff
        print(f'f1分数是：{ff}')
    overall_progress.update(1)
end_time=time.time()
print(f'F1_score{f1/200} time:{float(end_time-start_time)}')
overall_progress.close()