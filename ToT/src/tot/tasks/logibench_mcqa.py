import os
import pandas as pd
from ..tasks.base import Task, DATA_PATH
from ..prompts.logibench_mcqa import *
from datasets import load_dataset
import re
import json
class Logibench_MCQA(Task):
    """
    Input (x)   : A sentence with an ambiguous pronoun
    Output (y)  : The correct referent of the pronoun
    Reward (r)  : 1 if the selected referent is correct, else 0
    """
    def __init__(self,step):
        """
        file: a CSV file containing Winograd Schema Challenge data
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'logibench/logicbench_mcqa.json')
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[0:200]  # 读取整个 JSON 文件
        self.steps = step  # 只需一步完成任务
        self.value_cache = {}
        self.stops = ['\n']  # 生成停止符号

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        row = self.data[idx]
        return f"Context: {row['context']}\nQuestion: {row['question']}\nOptions:\nchoice_1:{row['choices']['choice_1']}\nchoice_2:{row['choices']['choice_2']}\nchoice_3:{row['choices']['choice_3']}\nchoice_4:{row['choices']['choice_4']}"


    def test_output(self, idx: int, output: str):
        correct_answer = self.data[idx]['answer']
        output=output.lower()
        print(f'output:{output}\n')
        match = re.search(r'\b(choice_1|choice_2|choice_3|choice_4)\b', output)  # 匹配 选项
        if match:
            predicted_answer = match.group(1)  # 提取匹配的 选项
            is_correct = int(predicted_answer == correct_answer.lower())  # 比对正确性
            return {'r': is_correct}
        else:
            print('not match yes/no\n')
            return {'r':0}

    # @staticmethod
    # def standard_prompt_wrap(x: str, y: str = '') -> str:
    #     return standard_prompt.format(input=x) + y
    #
    # @staticmethod
    # def cot_prompt_wrap(x: str, y: str = '') -> str:
    #     return cot_prompt.format(input=x) + y

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(input=x, current_steps=y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # 提取GPT返回的最后一行，应该是 'correct' 或 'incorrect'
        print(value_outputs)
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