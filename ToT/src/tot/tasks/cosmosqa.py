import os
import pandas as pd
from ..tasks.base import Task, DATA_PATH
from ..prompts.cosmosqa import *
from datasets import load_dataset
import re
import json
class Cosmos(Task):
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
        path = os.path.join(DATA_PATH, 'cosmos/train.csv')
        ds= load_dataset('csv', data_files=path)['train']
        # ds = ds.select(range(100))
        ds = ds.select(range(min(200, len(ds))))
        self.data=ds
        self.steps = step  # Number of reasoning steps.
        self.value_cache = {}
        self.stops = ['\n']  # Generation stop markers.

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        row = self.data[idx]
        return f"Context:\n{row['context']}\nQuestions: {row['question']}\nOptions:\nA:{row['answer0']}\nB:{row['answer1']}\nC:{row['answer2']}\nD:{row['answer3']}\n"


    # def test_output(self, idx: int, output: str):
    #     correct_answer_id = self.data[idx]['label']
    #     label_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    #     correct_answer=label_mapping[correct_answer_id]
    #     output=output.upper()
    #     print(f'output:{output}\n')
    #     match = re.search(r'\b(A|B|C|D)\b', output)  # Match answer options.
    #     if match:
    #         predicted_answer = match.group(1)  # Extract the matched option.
    #         is_correct = int(predicted_answer == correct_answer)  # Compare correctness.
    #         return {'r': is_correct}
    #     else:
    #         print('not match yes/no\n')
    #         return {'r':0}
    def test_output(self, idx: int, output: str):
        correct_answer_id = self.data[idx]['label']
        label_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_answer=label_mapping[correct_answer_id]
        output = output.strip().upper()
        print(f'output:{output}\n')

        # Use the final output line as the answer line.
        predict_answer = output.strip().split('\n')[-1]

        # If the final line contains a colon, keep the text after it.
        if ':' in predict_answer:
            predict_answer = predict_answer.split(':')[-1].strip()
        print(f"predict_answer after processing:{predict_answer}\n")
        match = re.search(r'\b(A|B|C|D|E)\b',predict_answer)  # Match answer options.
        if match:
            predict_answer = match.group(1)  # Extract the matched option.
            print(f"correct_answer:{correct_answer}, predict_answer:{predict_answer}\n")
            is_correct = int(predict_answer == correct_answer)  # Compare correctness.
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
        # Read the final judgment token from each model output.
        print(value_outputs)
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
