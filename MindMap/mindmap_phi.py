import numpy as np
import re
import string
from metric import F1_scorer
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from sklearn.metrics.pairwise import cosine_similarity
import sys
from time import sleep
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,help="Name of the dataset")
args = parser.parse_args()

LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/phi-3.8b"  # 本地模型路径或 Hugging Face 模型名称
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True)
llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

import re

import re

def extract_f1_answers(text):
    # 匹配 Output 1: 后面接内容或换行后的内容
    pattern = r'Output 1:\s*(?:"|“)?(.+?)(?:"|”)?(?:\n|$)'
    match = re.search(pattern, text)
    
    if match:
        answer = match.group(1).strip()
        # 如果内容为空，可能答案在下一行
        if not answer:
            # 获取 Output 1: 所在行的下一行
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("Output 1:"):
                    if i + 1 < len(lines):
                        return lines[i + 1].strip().strip('"“”')
        return answer.strip('"“”')
    return ''

def extract_first_option_letter(text):
    # 提取 Output1 或 Output 1 后面的第一行
    match = re.search(r'Output ?1:\s*(.+)', text)
    if match:
        first_line = match.group(1).strip()
        # 直接匹配第一个 A–E 大写字母
        letter_match = re.search(r'[A-E]', first_line)
        if letter_match:
            return letter_match.group(0)
    return ''
def extract_first_choice_letter(text):
    # 提取 Output1 或 Output 1 后面的第一行
    label={"A":"choices_1","B":"choices_2","C":"choices_3","D":"choices_4"}
    match = re.search(r'Output ?1:\s*(.+)', text)
    if match:
        first_line = match.group(1).strip()
        # 直接匹配第一个 A–E 大写字母
        letter_match = re.search(r'[A-E]', first_line)
        if letter_match:
            return label[letter_match.group(0)]
    return ''
def extract_fujia_data_letter(text):
    # 尝试匹配 "1:" 后面直接跟内容的情况
    match_inline = re.search(r"1:\s*(\S.*)", text)
    if match_inline:
        return match_inline.group(1).strip().lower()

    # 尝试匹配 "1:" 在行末的情况，然后取下一行
    match_nextline = re.search(r"1:\s*(?:\r?\n)(.*)", text)
    if match_nextline:
        return match_nextline.group(1).strip().lower()

    # 如果都没匹配到 "1:"，取最后一行
    last_line = text.strip().splitlines()[-1]
    return last_line.strip().lower()


def generate_response(prompt,):
    messages =[{"role": "user", "content": prompt}]
    text = llama_tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = llama_tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = llama_model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=4000,pad_token_id=llama_tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response_text = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response_text

# 这段代码的作用是：在 Neo4j 知识图谱中寻找两个实体之间的最短路径。
def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    entity_label = f"Entity_{args.dataset}"
    print(start_entity)
    print(end_entity)
    ss = (
    "MATCH (start_entity:" + entity_label + " {name: $start_entity_name}), "
    "(end_entity:" + entity_label + " {name: $end_entity_name}) "
    "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
    "RETURN p"
)
    with driver.session() as session:
        result = session.run(
            ss,
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]
        print(f'exist_entity:{exist_entity}\n')
        return paths,exist_entity

#这段代码的 作用 是将传入的多个列表进行 笛卡尔积运算，然后将所有生成的组合 扁平化为普通列表
#list1 = [1, 2]
# list2 = ['a', 'b']
#[[1, 'a'], [1, 'b'], [2, 'a'], [2, 'b']]
def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results

"""
这段代码的核心功能是 查询一个实体的所有邻居节点及其关系，并根据条件进行过滤与分类。
可以用于 构建知识图谱中的实体网络 或 提取特定类别的实体信息（如与疾病相关的实体）。
"""
def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
    query = f"""
    MATCH (e:`Entity_{args.dataset}`)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease


"""
这段代码的作用是：将知识图谱中的路径信息转换为自然语言表述。
"""
def prompt_path_finding(path_input):
    template = f"""
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {path_input}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    response_of_KG_path=generate_response(template)
    print(f'将知识图谱中的路径信息转换为自然语言描述：{response_of_KG_path}\n')
    return response_of_KG_path


"""
这段代码的作用是 将知识图谱中实体及其邻接关系转换为自然语言描述。它是基于用户输入的邻接信息（即实体间的关系）生成自然语言表述。使用的是 LangChain 和 ChatGPT 接口的方式来处理输入和生成输出。
"""
def prompt_neighbor(neighbor):
    template = f"""
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """
    response_of_KG_neighbor=generate_response(template)
    print(f'将知识图谱中的实体及其邻接关系转换为自然语言描述：{response_of_KG_neighbor}\n')
    return response_of_KG_neighbor


"""
这段代码计算的是 余弦相似度（Cosine Similarity），它是衡量两个向量在空间中方向相似度的常用方法。具体来说，这段代码计算的是 两个向量之间的余弦相似度，并通过手动实现的方式来计算。
"""
def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


"""
这段代码的作用是 自动换行文本，它根据给定的字体和最大宽度将长文本分割成多行，确保每行的宽度不超过指定的最大宽度。
"""
def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_answer(str,response_of_KG_list_path,response_of_KG_neighbor):
    messages  = [
                SystemMessage(content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "),
                HumanMessage(content="Patient input:"+ input_text[0]),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                HumanMessage(content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                            + "Output1: The answer includes disease and tests and recommened medications.\n\n"
                             +"Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                             +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                             + "There is a sample:\n"
                             + """
Output 1:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

Output 2:
Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').

Output 3: 
Patient(Path-based Evidence 1)
└── has been experiencing(Path-based Evidence 1)
    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
        └── could be caused by(Path-based Evidence 2)
            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                ├── requires(Neighbor-based Evidence 1)
                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                │       └── may include(Neighbor-based Evidence 2)
                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                ├── can be treated with(Path-based Evidence 3)
                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                └── should be accompanied by(Neighbor-based Evidence 3)
                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
                                    """
                             )

                                   ]
        
    result = chat(messages)
    output_all = result.content
    return output_all

"""
utput1：直接回答问题。例如，查询世界上最高的山是哪个，直接给出答案“珠穆朗玛峰”。

Output2：描述推理过程，如何通过知识图谱中的路径（例如，珠穆朗玛峰位于喜马拉雅山脉，并且是世界上最高的山）得出答案。

Output3：绘制推理过程的决策树，展示每个实体、关系以及证据来源（路径证据）。
"""
def final_answer_commonsense_knowledge(input_text, response_of_KG_list_path, response_of_KG_neighbor, options,context=None):
    # 定义Prompt的模板
    
    if args.dataset=="commonsense":
        prompt= f"""
        You are an AI model with a comprehensive general knowledge base. You are capable of answering questions on a wide range of topics such as science, geography, history, technology, culture, and more. 
        Use the provided knowledge graph to reason step by step and deliver precise answers.

        Follow the reasoning exactly to the output form of the given example

        User query: {input_text}

        The following general knowledge is available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Please answer the user's query by choosing one of the following options:
            A.{options[0]}
            B.{options[1]}
            C.{options[2]}
            D.{options[3]}
            E.{options[4]}

        Output 1: Provide a direct, factual answer based on general knowledge. Select an option from the list A,B,C,D,E, and output the option letter in uppercase.
        
        Output 2: Show the reasoning process step by step, using knowledge graph paths and neighbors to explain how you arrive at the answer.
        
        Output 3: Draw a decision tree showing the inference process. Include entities, relationships, and evidence sources in the tree.
        
        Follow the reasoning exactly to the output form of the given example
        Example:

        Output 1:
        The tallest mountain in the world is Mount Everest, located in the Himalayas on the border between Nepal and China. Its height is approximately 8,848.86 meters above sea level. The correct answer is: A) Mount Everest.
        
        Output 2:
        Path-based Evidence 1('Mount Everest'->'is located in'->'Himalayas')->Path-based Evidence 2('Mount Everest'->'is the tallest mountain'->'world')->result 1('Mount Everest').
        
        Output 3:
        Mount Everest(Path-based Evidence 1)
        └── is located in(Path-based Evidence 1)
            └── Himalayas(Path-based Evidence 1)
                └── is the tallest mountain(Path-based Evidence 2)
                    └── world(Path-based Evidence 2)(result 1)
        """
    if args.dataset=="cosmosqa":
        prompt = f"""
        You are an AI model with a comprehensive general knowledge base. You are capable of answering questions on a wide range of topics such as science, geography, history, technology, culture, and more. 
        Use the provided knowledge graph and the given context to reason step by step and deliver precise answers.

        Context: {context}

        User query: {input_text}

        The following general knowledge is available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Please answer the user's query by choosing one of the following options:
            A.{options[0]}
            B.{options[1]}
            C.{options[2]}
            D.{options[3]}

        Output 1: Provide a direct, factual answer based on general knowledge. Select an option from the list A,B,C,D, and output the option letter in uppercase.
        
        Output 2: Show the reasoning process step by step, using the context, knowledge graph paths, and neighbors to explain how you arrive at the answer.
        
        Output 3: Draw a decision tree showing the inference process. Include the context, entities, relationships, and evidence sources in the tree.
        
        Example:

        Output 1:
        Based on the context provided, the tallest mountain in the world is Mount Everest, located in the Himalayas on the border between Nepal and China. Its height is approximately 8,848.86 meters above sea level. The correct answer is: A) Mount Everest.
        
        Output 2:
        Context-based Evidence 1 (Context mentions 'Mount Everest' and its location 'Himalayas') -> 
        Path-based Evidence 2 ('Mount Everest'->'is the tallest mountain'->'world') -> 
        Result 1 ('Mount Everest').
        
        Output 3:
        Context (Evidence Source)
        └── Mentions ('Mount Everest' and 'Himalayas')
            └── is the tallest mountain (Path-based Evidence 2)
                └── world (Path-based Evidence 2) (result 1)
        """
    
    if args.dataset=="winograd":
        prompt = f"""
        You are an AI model with a comprehensive general knowledge base. You are capable of answering questions on a wide range of topics such as language understanding, coreference resolution, and reasoning.

        Use the provided context and knowledge graph information to reason step by step and deliver precise answers about pronoun resolution.

        Context: {context}

       {input_text}

        The following general knowledge from the knowledge graph is available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Options:
        {options[0]}
        {options[1]}


        Output 1: Provide a direct, factual answer by naming the exact option (from above) that the pronoun refers to.

        Output 2: Show the reasoning process step by step, using the context sentence, the pronoun phrase, and relevant knowledge graph information to explain how you arrive at the answer.

        Output 3: Draw a decision tree illustrating the inference process. Include the context, entities (options), relationships, pronoun phrase, and evidence sources in the tree.

        Example:

        Output 1:
        The pronoun 'he' most likely refers to 'John'.

        Output 2:
        - The pronoun 'he' appears in the phrase 'he picked up the book', indicating the subject performing the action.
        - Among the options, 'John' is mentioned in the context as someone who was tired and then picked up the book.
        - 'Mike' is not mentioned in the context or related to the action.
        - Knowledge graph paths show 'John' is associated with 'reads' and 'book'.
        - Therefore, the pronoun 'he' most likely refers to 'John'.
        ...

        Output 3:
        Context (Sentence)
        └── Pronoun phrase: 'he picked up the book'
            ├── Option: 'John' (mentioned as subject performing action)
                └── Related to 'reads' and 'book' in KG evidence
            └── Option: 'Mike' (no relevant evidence)
        ...
        """
    if args.dataset=="bqa":
         prompt = f"""
            You are an AI model with strong logical reasoning and commonsense inference capabilities. 
            You can analyze provided contexts, interpret statements, and determine whether certain conclusions follow logically.

            Use the provided context and knowledge graph information to reason step by step and deliver precise answers to the question.

            Context:
            {context}

            Question:
            {input_text}

            The following general knowledge from the knowledge graph is available:
            ### {response_of_KG_list_path}
            ### {response_of_KG_neighbor}

            Output 1: Provide a direct 'yes' or 'no' answer to the question based purely on logical reasoning and the given context.

            Output 2: Show the reasoning process step by step, using:
            - The logical structure of the statements in the context
            - The question’s phrasing and its logical implications
            - Relevant knowledge graph information (if applicable)
            Explain how each piece of evidence leads to your conclusion.

            Output 3: Draw a logic diagram or decision tree illustrating the reasoning process. 
            Include:
            - Context statements
            - Key variables or propositions
            - Logical relations (AND, OR, NOT, IF-THEN)
            - The path from premises to the final answer

            Example:

            Output 1:
            no

            Output 2:
            - Context says: "At least one of A or B is true."
            - Question asks: "Can we say that C or D must be true?"
            - There is no logical connection ensuring that C or D is always true.
            - Therefore, the answer is "no".

            Output 3:
            Context
            └── Proposition A: "Tom studies diligently"
            └── Proposition B: "Tom excels academically"
                ├── Rule: At least one of A or B is true
                └── Question propositions: C = "He won't get good grades", D = "He watches too much TV"
            Logical check: No necessity link from A/B to C/D → Answer: no
            """
    if args.dataset=="mcqa":
         prompt = f"""
            You are an AI model with strong logical reasoning and commonsense inference capabilities.
            You can analyze provided contexts, evaluate multiple-choice options, and determine the most logically suitable conclusion.

            Use the provided context and knowledge graph information to reason step by step and choose the single best answer from the given options.

            Context:
            {context}

            Question:
            {input_text}

            The following general knowledge from the knowledge graph is available:
            ### {response_of_KG_list_path}
            ### {response_of_KG_neighbor}

            Options:
            choice_1: {options["choice_1"]}
            choice_2: {options["choice_2"]}
            choice_3: {options["choice_3"]}
            choice_4: {options["choice_4"]}

            Output 1: Provide the direct answer only as the option key (e.g., "choice_3") without explanation.

            Output 2: Show the reasoning process step by step, including:
            - The logical interpretation of the statements in the context
            - How each option aligns or conflicts with the given context
            - The role of any relevant knowledge graph facts
            - Why the chosen option is more logically correct than the others

            Output 3: Draw a decision tree or logic diagram illustrating the reasoning process.
            Include:
            - Key propositions from the context
            - Logical operators (AND, OR, NOT, IF-THEN)
            - Evaluation path for each option
            - Final decision

            Example:

            Output 1:
            choice_3

            Output 2:
            - Context says: At least one of ("Jane consumes ample water" OR "She will not experience a sugar crash") is true.
            - Option 3 restates this idea as "She will feel hydrated" (from ample water) OR "She doesn't eat too much sugar" (preventing sugar crash).
            - The other options introduce unrelated subjects or incorrect logical relationships.
            - Therefore, Option 3 is the most logically consistent with the context.

            Output 3:
            Context
            └── Proposition A: Jane consumes ample water → hydrated
            └── Proposition B: She will not experience a sugar crash
                ├── Rule: At least one of A or B is true
            Options evaluation:
                choice_1: Conflicts with given rules → reject
                choice_2: Adds unrelated subject (John) → reject
                choice_3: Matches A OR B → accept 
                choice_4: Logical negation of A OR B → reject
            """
    
    if args.dataset=="sciq":
        prompt = f"""
        You are an AI model with a comprehensive scientific knowledge base. You are capable of answering questions on a wide range of scientific topics such as physics, chemistry, biology, astronomy, mathematics, and more. 
        Use the provided knowledge graph to reason step by step and deliver precise, evidence-based answers.

        User query: {input_text}

        The following scientific knowledge is available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Please answer the user's query by choosing one of the following options:
            A.{options[0]}
            B.{options[1]}
            C.{options[2]}
            D.{options[3]}

        Output 1: Provide a direct, factual answer based on general knowledge. Select an option from the list A,B,C,D, and output the option letter in uppercase.
       
        Output 2: Show the reasoning process step by step, using knowledge graph paths and neighbors to explain how you arrive at the answer.
        
        Output 3: Draw a decision tree showing the inference process. Include scientific concepts, relationships, and evidence sources in the tree.
        
        Example:

        Output 1:
        The chemical formula for water is H2O, which consists of two hydrogen atoms and one oxygen atom. The correct answer is: A. H2O.
        
        Output 2:
        Path-based Evidence 1('Water'->'consists of'->'Hydrogen and Oxygen')->Path-based Evidence 2('H2O'->'represents'->'two Hydrogen atoms and one Oxygen atom')->result 1('H2O').
        
        Output 3:
        Water (Path-based Evidence 1)
        └── consists of (Path-based Evidence 1)
            └── Hydrogen and Oxygen (Path-based Evidence 1)
                └── H2O (Path-based Evidence 2)
                    └── represents (Path-based Evidence 2)
                        └── Two Hydrogen atoms and one Oxygen atom (result 1)
        """
    if args.dataset=="medqa":
        prompt = f"""
        You are an AI model with a comprehensive medical knowledge base. You are capable of answering questions on a wide range of topics such as human anatomy, physiology, diseases, treatments, medications, diagnostics, and healthcare. 
        Use the provided medical knowledge graph to reason step by step and deliver accurate, evidence-based answers.

        User query: {input_text}

        The following medical knowledge is available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Please answer the user's query by choosing one of the following options:
            A.{options["A"]}
            B.{options["B"]}
            C.{options["C"]}
            D.{options["D"]}
            E.{options["E"]}

        Output 1: Provide a direct, factual answer based on general knowledge. Select an option from the list A,B,C,D,E, and output the option letter in uppercase.
        
        Output 2: Show the reasoning process step by step, using knowledge graph paths and neighbors to explain how you arrive at the answer.
        
        Output 3: Draw a decision tree showing the inference process. Include medical entities (e.g., symptoms, diseases, treatments), relationships, and evidence sources in the tree.
        
        Example:

        Output 1:
        Hypertension is often treated with medications such as ACE inhibitors, beta-blockers, or diuretics. The correct answer is: C) ACE inhibitors.

        Output 2:
        Path-based Evidence 1('Hypertension'->'can be treated with'->'ACE inhibitors')->Path-based Evidence 2('ACE inhibitors'->'reduce'->'blood pressure')->result 1('ACE inhibitors').

        Output 3:
        Hypertension (Path-based Evidence 1)
        └── can be treated with (Path-based Evidence 1)
            └── ACE inhibitors (Path-based Evidence 1)
                └── reduce (Path-based Evidence 2)
                    └── blood pressure (result 1)
        """

    if args.dataset == "squad":
        prompt = f"""
        You are an AI model with a comprehensive general and factual knowledge base. You are capable of answering questions based on both the provided context and structured knowledge. 
        Use the context passage and the knowledge graph to reason step by step and deliver a concise, precise answer.

        Context:
        {context}

        User query:
        {input_text}

        The following general knowledge is available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Output 1: Provide only the final answer. The answer should be concise (preferably a short span or phrase from the context), and suitable for F1 evaluation. Do not include any explanation or extra words — just return the final answer.

        Output 2: Show the reasoning process step by step, using context and knowledge graph paths/neighbors to explain how you arrived at the answer.

        Output 3: Draw a reasoning tree showing the inference process. Include context-derived entities, knowledge relationships, and sources of supporting evidence.

        Example:

        Output 1:
        Mount Everest

        Output 2:
        Context-based Evidence 1 (The context states that "Mount Everest is the tallest mountain in the world.") ->
        KG Path 1 ('Mount Everest' -> 'is located in' -> 'Himalayas') ->
        Result: Mount Everest.

        Output 3:
        Context (Evidence Source)
        └── Mentions 'Mount Everest'
            └── is the tallest mountain (from context)
                └── supported by KG ('Mount Everest' -> 'located in' -> 'Himalayas')
                    └── Result: Mount Everest
        """
    if args.dataset in "hotpotqa 2multiwiki":
        prompt = f"""
        You are an advanced AI model capable of answering complex multi-hop questions based on multiple context paragraphs and supporting knowledge graph facts.

        Use the provided context (which may span multiple paragraphs) and the external knowledge graph to reason step by step and extract the correct answer.

        Context:
        {context}   # 包含多个段落

        User query:
        {input_text}

        The following knowledge graph facts are available:
        ### {response_of_KG_list_path}
        ### {response_of_KG_neighbor}

        Output 1: Provide the final answer. It should be concise (a short span or phrase), ideally from the context, and optimized for F1 evaluation. No explanation, just the final answer.

        Output 2: Step-by-step reasoning using both the multi-hop context and the knowledge graph. Explain how each piece of evidence leads to the final answer.

        Output 3: A reasoning tree or chain that shows how facts from multiple paragraphs and KG entries are combined to arrive at the answer.

        Example:

        Output 1:
        Chicago Bulls

        Output 2:
        Para 1 mentions that Michael Jordan played for the Chicago Bulls. 
        Para 2 discusses his role in winning six NBA championships.
        KG path confirms ('Michael Jordan' -> 'played for' -> 'Chicago Bulls').
        Combined, this supports the answer: Chicago Bulls.

        Output 3:
        Michael Jordan (from Q)
        ├── played for (KG + Para 1)
        │   └── Chicago Bulls
        └── won championships (Para 2)
            └── supports team identity (Chicago Bulls)
        → Result: Chicago Bulls
        """

    # 使用LLaMA模型进行推理
    result=generate_response(prompt)
    print(f"final_answer:{result}\n")
    # 返回完整的结果
    return result

"""
这段代码的作用是根据传入的 问题 (question) 和 医疗知识信息 (instruction)，生成一个完整的模板并将其格式化，然后利用该模板向 AI 模型请求一个响应，最终返回模型的回答。以下是详细的步骤和解释：
"""

if __name__ == "__main__":
    start_time=time.time()
    exist_entity = None
    # 1. build neo4j knowledge graph datasets
    uri = "bolt://10.149.9.30:7687"
    username = "neo4j"
    password = "liujie&931"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()


    ##############################build KG 

    # session.run("MATCH (n) DETACH DELETE n")# clean all

    
    # 读取 CSV 文件（确保文件路径正确）
    df = pd.read_csv(f'./dataset/{args.dataset}/relation.csv')  # CSV 文件中第一行为列名：Entity1,Relation,Entity2
    context=None
    # 遍历每一行，创建节点和关系
    for index, row in df.iterrows():
        head_name = row['Entity1']
        relation_name = row['Relation']
        tail_name = row['Entity2']
        # 根据数据集动态设置标签或关系类型
        node_label_head = f"Entity_{args.dataset}"
        node_label_tail = f"Entity_{args.dataset}"
        relation_type = f"{relation_name}_{args.dataset}"
        
        query = (
            "MERGE (h:`" + node_label_head + "` { name: $head_name }) "
            "MERGE (t:`" + node_label_tail + "` { name: $tail_name }) "
            "MERGE (h)-[r:`" + relation_type + "`]->(t)"
        )
        session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)



    with open(f'./dataset/{args.dataset}/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open(f'./dataset/{args.dataset}/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)
    correct=0
    F1=0.
    with open(f"./dataset/{args.dataset}/keyword.json", "r",encoding="utf-8") as f:
        json_data = json.load(f)
        overall_progress = tqdm(total=len(json_data), desc="Overall Progress", unit="task")
        for line in json_data:
            keywords = line.get("keywords")
            question=line.get("question")
            context=None
            if args.dataset in "hotpotqa 2multiwiki squad bqa mcqa winograd":
                context=line.get("context")
            if args.dataset=='commonsense':
                options=line.get("options", {}).get("text", [])
            else:
                options=line.get("options", {})
            answer=line.get("answer")
            if args.dataset=='cosmosqa':
                context=line.get("context")
            question_kg = [kw.strip() for kw in keywords.split(",") if kw.strip()]
            # print("question_kg",question_kg)

            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
           

            for kg_entity in question_kg:
                
                try:
                    keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                    print(f'KG_Enentty:{kg_entity}')
                except ValueError:
                        print(f"[Warning]-------- Keyword '{kg_entity}' not found in keyword embeddings, skipping.")
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ","_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ","_"))
            # print('match_kg',match_kg)

            # # 4. neo4j knowledge graph path finding
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)                        
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    result_path = combine_lists(*paths_list)
                
                
                    if result_path != []:
                        result_path_list.extend(result_path)                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                start_tmp = []
                for path_new in result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        result_path = {}
                        single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                                                  
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                result_path = {}
                single_path = {}            
            # print('result_path',result_path)
            
            

            # # 5. neo4j knowledge graph neighbor entities
            neighbor_list = []
            neighbor_list_disease = []
            for match_entity in match_kg:
                disease_flag = 0
                neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
                neighbor_list.extend(neighbors)

                while disease != []:
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in match_kg:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            neighbor_list_disease.extend(neighbors)
                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                            neighbor_list_disease.extend(neighbors)
            if len(neighbor_list)<=5:
                neighbor_list.extend(neighbor_list_disease)

            # print("neighbor_list",neighbor_list)


            # 6. knowledge gragh path based prompt generation
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                    
                    path = "\n".join(result_new_path)
                    response_of_KG_list_path = prompt_path_finding(path)
                    # print("response_of_KG_list_path",response_of_KG_list_path)
            else:
                response_of_KG_list_path = '{}'

            response_single_path = prompt_path_finding(single_path)

            # # 7. knowledge gragh neighbor entities based prompt generation   
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

           
            neighbor_input = "\n".join(neighbor_new_list[:5])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input)
            # print("response_of_KG_neighbor",response_of_KG_neighbor)


            # # 8. prompt-based medical diaglogue answer generation
            # output_all = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_neighbor)
            output_all=final_answer_commonsense_knowledge(question,response_of_KG_list_path,response_of_KG_list_neighbor,options,context)
            # 使用正则表达式提取 Output 1 与 Output 2 之间的文本
            match = re.search(r"Output 1:\n(.*?)\nOutput 2:", output_all, re.DOTALL)

            if args.dataset in "commonsense sciq medqa cosmosqa bqa mcqa winograd":
                if args.dataset =="sciq":
                    predict=extract_first_choice_letter(output_all)
                else:
                    predict=extract_first_option_letter(output_all)

                #上面的正则提取对于附加的三个逻辑数据集不起作用
                #!!!!!
                if args.dataset in " mcqa bqa winograd":
                    predict=extract_fujia_data_letter(output_all)
                    print(f'predict:{predict}\n')
                    print(f'answer:{answer}\n')
                    #注意这里 这三个补充数据集 是全小写比对 
                    if(answer.lower() in predict.lower()):
                        print("答案正确！\n")
                        correct+=1
                    else:
                        print("答案错误！")
                else:
                    # 这里的数据集 涉及到ABCD 不能用小写比对 正则
                    print(f'predict:{predict}\n')
                    print(f'answer:{answer}\n')
                    if(answer in predict):
                        print("答案正确！\n")
                        correct+=1
                    else:
                        print("答案错误！")  
              
            else:
                predict=extract_f1_answers(output_all)
                print(f'predict:{predict}\n')
                print(f'answer:{answer}\n')
                ff=F1_scorer([predict],[answer])
                F1+=ff
                print(f'now_f1:{ff}  sum_f1:{F1}\n')
            overall_progress.update(1)
    end_time=time.time()
    cost=float(end_time-start_time)
    print(f'time:{cost:.2f} s')
    print(f'F1:{F1/200}\n')
    print(f'ACC:{correct/200}')

           
                
               