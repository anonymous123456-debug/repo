import subprocess
from datasets import load_dataset
import time
from tqdm import tqdm
import re
import json
from metric import F1_scorer

def extract_final_option(text):
    result = re.search(r"final_answer:\s*(.*)", text)
    if result:
        extracted_text = result.group(1).strip()
        return extracted_text
    else:
        print("not match")
        return "not match"


with open(f'../dataset/hotpotqa.json', encoding='utf-8') as f:
    ds = json.load(f)
    ds=ds[:200]

f1=0.
overall_progress = tqdm(total=len(ds), desc="Overall Progress", unit="task")
start_time=time.time()
for item in ds:
    quesiton=item['question']
    answer=item['answer']
    query_text = f"""
    Question:
        {quesiton}
    final_answer:

    Please answer the above question, at the end of your reasoning, you only need to output a short answer,Only give me the answer and do not output any other words.Just write a short answer, not a long one.
    The answer format is :final_answer:
    Follow the reasoning exactly to the output form of the given example
    
    Example
        Question:Which 47th Governor of Indiana beat Republican David M. McIntosh in 2000?
        final_answer:Frank O'Bannon
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