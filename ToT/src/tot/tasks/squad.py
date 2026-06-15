import os
import pandas as pd
from ..tasks.base import Task, DATA_PATH
from ..prompts.squad import *
from datasets import load_dataset
from .metric import F1_scorer
import re
import json
class SquadTask(Task):
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
        path = os.path.join(DATA_PATH, 'squad/squad.json')
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)[:200]  # Load the first 200 samples.
        self.steps = step # Number of reasoning steps.
        self.value_cache = {}
        self.stops = ['\n']  # Generation stop markers.

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        row = self.data[idx]
        return f"Context:\n{row['context']}\nQuestion:\n{row['question']}"

    def test_output(self, idx: int, output: str):
        correct_answer = self.data[idx]['answers']['text'][0]
        output = output.lower()
        print(f'output:{output}\n')
        f1_score = F1_scorer([output], [correct_answer])  # Compute the F1 score.
        return {'f1': f1_score}

        
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(input=x, current_steps=y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # Read the final judgment token from each model output.
        value_names = [re.sub(r'[^a-z]', '', _.strip().lower().split()[-1]) for _ in value_outputs]
        print(f'value_names:{value_names}\n')
        # Map qualitative judgments to scores.
        value_map = {'best': 5, 'good': 2.5,'bad':0.5}

        # Average the judgment scores.
        value = sum(value_map.get(name, 0) for name in value_names) / len(value_names)
        return value

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        return propose_prompt.format(input=x,current_steps=y)

    @staticmethod
    def propose_final_prompt_wrap(x: str, y: str = '') -> str:
        return last_prompt.format(input=x,reasoning_steps=y)
    
    
