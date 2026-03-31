import os
import pandas as pd
from ..tasks.base import Task, DATA_PATH
from ..prompts.WikiMultiHopQA import *
from datasets import load_dataset
from .metric import F1_scorer
import re
import json
class WikiMultiHopQATask(Task):
    """
    Input (x)   : A sentence with an ambiguous pronoun
    Output (y)  : The correct referent of the pronoun
    Reward (r)  : 1 if the selected referent is correct, else 0
    """
    def __init__(self, step):
        """
        file: a CSV file containing Winograd Schema Challenge data
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '2WikiMultiHopQA/2WikiMultiHopQA.json')
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[:200]  # 读取整个 JSON 文件
        self.steps = step # 只需一步完成任务
        self.value_cache = {}
        self.stops = ['\n']  # 生成停止符号

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        row = self.data[idx]
        return f"Context:\n{row['context']}\nQuestion:\n{row['question']}"

    def test_output(self, idx: int, output: str):
        correct_answer = self.data[idx]['answer']
        output = output.lower()
        print(f'output:{output}\n')
        f1_score = F1_scorer([output], [correct_answer])  # 计算 F1 分数
        return {'f1': f1_score}

        
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(input=x, current_steps=y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # 提取GPT返回的最后一行，应该是 'correct' 或 'incorrect'
        value_names = [re.sub(r'[^a-z]', '', _.strip().lower().split()[-1]) for _ in value_outputs]
        print(f'value_names:{value_names}\n')
        # 映射 'correct' 和 'incorrect' 到分数
        value_map = {'best': 5, 'good': 2.5,'bad':0.5}

        # 计算评分（正确数目占比）
        value = sum(value_map.get(name, 0) for name in value_names) / len(value_names)
        return value

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        return propose_prompt.format(input=x,current_steps=y)

    @staticmethod
    def propose_final_prompt_wrap(x: str, y: str = '') -> str:
        return last_prompt.format(input=x,reasoning_steps=y)