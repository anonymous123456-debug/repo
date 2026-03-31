import os
import pandas as pd
from ..tasks.base import Task, DATA_PATH
from ..prompts.sciq import *
from datasets import load_dataset
import re
import json
class SciqTask(Task):
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
        path = os.path.join(DATA_PATH, 'sciq/sciq_2000.json')
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[0:200]  # 读取整个 JSON 文件
        self.steps = step # 只需一步完成任务
        self.value_cache = {}
        self.stops = ['\n']  # 生成停止符号

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        row = self.data[idx]
        return f"Question:\n{row['question']}\nOptions:\nA:{row['choices']['A']}\nB:{row['choices']['B']}\nC:{row['choices']['C']}\nD:{row['choices']['D']}\n"

    # def test_output(self, idx: int, output: str):
    #     correct_answer = self.data[idx]['correct_answer']
    #     output = output.upper()
    #     print(f'output:{output}\n')
    #     match = re.search(r'\b(A|B|C|D)\b', output)  # 匹配 选项
    #     if match:
    #         predicted_answer = match.group(1)  # 提取匹配的 选项
    #         is_correct = int(predicted_answer == correct_answer)  # 比对正确性
    #         return {'r': is_correct}
    #     else:
    #         print('not match ABCDE\n')
    #         return {'r':0}
    def test_output(self, idx: int, output: str):
        correct_answer = self.data[idx]['correct_answer']
        output = output.strip().upper()
        print(f'output:{output}\n')

        # 获取最后一行
        predict_answer = output.strip().split('\n')[-1]

        # 如果最后一行包含冒号，则取冒号后的内容
        if ':' in predict_answer:
            predict_answer = predict_answer.split(':')[-1].strip()
        print(predict_answer)
        match = re.search(r'\b(A|B|C|D|E)\b',predict_answer)  # 匹配 选项
        if match:
            predict_answer = match.group(1)  # 提取匹配的 选项
            is_correct = int(predict_answer == correct_answer)  # 比对正确性
            return {'r': is_correct}
        else:
            print('not match yes/no\n')
            return {'r':0}
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