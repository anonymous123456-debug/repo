from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset

import torch
# 本地 LLaMA 配置
LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/servy/llama3-8b"
# LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/qwen-1.5b"  # 本地模型路径或 Hugging Face 模型名称
# LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/smolLM2-1.7b"  # 本地模型路径或 Hugging Face 模型名称
# LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/phi-3.8b"  # 本地模型路径或 Hugging Face 模型名称
# LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/qwen2.5-7b"  # 本地模型路径或 Hugging Face 模型名称
# LLAMA_MODEL_PATH = "/share/home/ncu_418000240001/mistral"  # 本地模型路径或 Hugging Face 模型名称

#
llama_tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_MODEL_PATH, trust_remote_code=True
)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations

#orgin
# def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
#     if "llama" in engine.lower():
#         openai.api_key = "EMPTY"
#         openai.api_base = "http://localhost:8000/v1"  # your local llama server port
#         engine = openai.Model.list()["data"][0]["id"]
#     else:
#         openai.api_key = opeani_api_keys
#
#     messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
#     message_prompt = {"role":"user","content":prompt}
#     messages.append(message_prompt)
#     f = 0
#     while(f == 0):
#         try:
#             response = openai.ChatCompletion.create(
#                     model=engine,
#                     messages = messages,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     frequency_penalty=0,
#                     presence_penalty=0)
#             result = response["choices"][0]['message']['content']
#             f = 1
#         except:
#             print("openai error, retry")
#             time.sleep(2)
#     return result
total_completion_tokens=0
total_prompt_tokens=0
def gpt_usage():
    global total_completion_tokens, total_prompt_tokens
    print({"completion_tokens": total_completion_tokens, "prompt_tokens": total_prompt_tokens})
    print(total_completion_tokens + total_prompt_tokens)

def generate_response(prompt: str, max_new_tokens: int = 4000) -> str:
    """
    使用本地 LLaMA 模型生成回复
    """
    global total_prompt_tokens ,total_completion_tokens
    messages = [{"role": "user", "content": prompt}]
    text = llama_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = llama_tokenizer([text], return_tensors="pt").to("cuda")
     # 统计 prompt token 数
    prompt_tokens = len(model_inputs.input_ids[0])
    total_prompt_tokens += prompt_tokens
    generated_ids = llama_model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        pad_token_id=llama_tokenizer.eos_token_id
    )
    # 去掉输入部分，只保留生成的内容
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
     # 统计 completion token 数
    completion_tokens = len(generated_ids[0])
    total_completion_tokens += completion_tokens
    response_text = llama_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return response_text



def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    """
    统一的 LLM 调用接口：
      - 如果 engine 包含 "gpt" 或 "deepseek"，走 OpenAI 接口
      - 否则走本地 LLaMA 模型
    """
    if "gpt" in engine.lower() or "deepseek" in engine.lower():
        openai.api_key = opeani_api_keys
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt}
        ]
        f = 0
        while f == 0:
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result = response["choices"][0]['message']['content']
                f = 1
            except Exception as e:
                print(f"openai error, retry: {e}")
                time.sleep(2)
        return result
    else:
        # 本地 llama 模型调用
        return generate_response(prompt, max_new_tokens=max_tokens)
    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open("./misral/ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

def process_result(question, pred_answer, cluster_chain_of_entities, gold_answer, results_list):
    """
    处理单个问题的结果，保存推理链、预测结果，并检查是否正确
    """
    is_correct = (pred_answer == gold_answer)
    results_list.append({
        "question": question,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "correct": is_correct,
        "reasoning_chains": cluster_chain_of_entities
    })
    return is_correct
    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    ####################ours
    elif dataset_name == 'commonsenseqa':
        with open('../data/commonsenseqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'cosmosqa':
        with open('../data/cosmosqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'sciq':
        with open('../data/sciq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'medqa':
        with open('../data/medqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'squad':
        with open('../data/squad.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'hotpotqa':
        with open('../data/hotpotqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== '2multiwiki':
        with open('../data/2WikiMultiHopQA.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'mcqa':
        with open('../data/mcqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'bqa':
        with open('../data/bqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name== 'winograd':
        with open('../data/winograd.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    ######################ours
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string