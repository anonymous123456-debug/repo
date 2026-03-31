import re
import json
import faiss 
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForSequenceClassification
from transformers.generation.utils import GenerationConfig
import numpy as np
import torch
import os
import random
from sentence_transformers import SentenceTransformer
from datetime import datetime
import backoff
import logging
import argparse
import yaml
from metric import F1_scorer
from api import call_api
from datasets import load_dataset
logger = logging.getLogger()

choices = [
    "glm-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125","chatGLM3-6b-32k", "chatGLM3-6b-8k","LongAlign-7B-64k",
    "qwen1.5-7b-chat-32k", "vicuna-v1.5-7b-16k","Llama3-8B-Instruct-8k", "Llama3-70b-8k", "Llama2-13b-chat-longlora",
    "LongRAG-chatglm3-32k", "LongRAG-qwen1.5-32k","LongRAG-vicuna-v1.5-16k", "LongRAG-llama3-8k",  "LongRAG-llama2-4k","qwen-1.5b","smolLM2-1.7b"
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "squad","cosmosqa","commonsenseqa","sciq","medqa","winograd","mcqa","bqa"], default="hotpotqa", help="Name of the dataset")
parser.add_argument('--top_k1', type=int, default=100, help="Number of candidates after initial retrieval")
parser.add_argument('--top_k2', type=int, default=20, help="Number of candidates after reranking")
parser.add_argument('--model', type=str, choices=choices, default="chatGLM3-6b-32k", help="Model for generation")
parser.add_argument('--lrag_model', type=str, choices=choices, default="", help="Model for LongRAG")
parser.add_argument('--rb', action="store_true", default=False, help="Vanilla RAG")
parser.add_argument('--raw_pred', action="store_true", default=False, help="LLM direct answer without retrieval")
parser.add_argument('--rl', action="store_true", default=False, help="RAG-Long")
parser.add_argument('--ext', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--raw_context_pred', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--fil', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--ext_fil', action="store_true", default=False, help="Using Extractor and Filter")
parser.add_argument('--MaxClients', type=int, default=1)
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('--r_path', type=str, default="../data/corpus/processed/200_2_2", help="Path to the vector database")
args = parser.parse_args()


def load_bqa_ds(filepath):
    result=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
         result.append({
                'context':d['context'],
                'answer': d['answer'],
                'question': d['question']
            })
    return result
def load_ds_mcqa(filepath):
    result=[]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
         result.append({
                'id':d['id'],
                'choices':d['choices'],
                'context':d['context'],
                'answer': d['answer'],
                'question': d['question']
            })
    return result

def get_word_len(input):
    return len(input)

def set_prompt(input, maxlen):
    if len(input) > maxlen:
        half = int(maxlen * 0.5)
        input = input[:half] + input[-half:]
    return input, len(input)

# prompt的长度
# def get_word_len(input):
#     tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
#     return len(tokenized_prompt)
# # 拼接 如果超出maxlen
# def set_prompt(input, maxlen):
#     tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
#     if len(tokenized_prompt) > maxlen:
#          half = int(maxlen * 0.5)
#          input = set_prompt_tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + set_prompt_tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
#     return input, len(tokenized_prompt)

# seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model2path, model_name):
    if "gpt" in model_name or "glm-4" in model_name or "glm3-turbo-128k" in model_name:
        return model_name, model_name
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name or "longalign-6b" in model_name or "qwen" in model_name or "llama3" or "qwen" or "smol" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model2path[model_name], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    # if "vicuna" in model_name:
    #     from fastchat.model import load_model
    #     model, _ = load_model(model2path[model_name], device="cpu", num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
    #     model = model.to(device)
    #     model = model.bfloat16()
    #     tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True, use_fast=False)
    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model2path[model_name])
        model = LlamaForCausalLM.from_pretrained(model2path[model_name], torch_dtype=torch.bfloat16, device_map='auto')
    model = model.eval()
    return model, tokenizer

@backoff.on_exception(backoff.expo, (Exception), max_time=200)
def pred(model_name, model, tokenizer, prompt, maxlen, max_new_tokens=32, temperature=1):
    try:
        if "longalign" in model_name.lower() and max_new_tokens == 32:
            max_new_tokens = 128
        prompt, prompt_len = set_prompt(prompt, maxlen)
        history = []
        if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
            response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature, num_beams=1, do_sample=False)
            return response, prompt_len
        elif "baichuan" in model_name:
            messages = [{"content": prompt, "role": "user"}]
            model.generation_config = GenerationConfig.from_pretrained(model2path["baichuan2-7b-4k"], temperature=temperature, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            response = model.chat(tokenizer, messages)
            return response, prompt_len
        elif "llama3" or "qwen" or "smol" in model_name:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=1000,pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response, prompt_len
        elif "glm-4" in model_name or "glm3-turbo-128k" in model_name or "gpt" in model_name:
            response = call_api(prompt, model_name, max_new_tokens)
            return response, prompt_len
        elif "qwen" in model_name:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, temperature=1.0)
            response = tokenizer.batch_decode([output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)], skip_special_tokens=True)[0]
            return response, prompt_len
        elif "llama" in model_name:
            input = tokenizer(f"[INST]{prompt}[/INST]", truncation=False, return_tensors="pt").to(model.device)
  
        context_length = input.input_ids.shape[-1]
        output = model.generate(**input, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, temperature=temperature)
        response = tokenizer.decode(output[0][context_length:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(1)
        return None, None
    return response, prompt_len



def setup_logger(logger, filename='log'):
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)

def print_args(args):
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = f"{key}{' ' * (30 - len(key))}"
        logger.info(f"{keystr} --> {val}")
    logger.info(f"LongRAG model used: {args.lrag_model}")
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")

def find_question_id(json_file, search_text):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 加载 JSON 数据

    for item in data:
        if search_text.lower() in item.get("question", "").lower():  # 忽略大小写匹配
            return item.get("question_id")  # 返回匹配的 question_id

    return None  # 未找到匹配项
def find_text_by_question_id(json_file, question_id):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 加载 JSON 数据
    if question_id is None:return None
    matching_texts = []
    for item in data:
        if item.get("question_id") == question_id:
            matching_texts.append(item.get("paragraph_text"))  # 收集所有匹配的 text 字段

    # 将匹配的 text 字段合并为一个字符串，使用换行符分隔
    return "\n".join(matching_texts) if matching_texts else None  # 如果找不到匹配的 text 返回 None

def search_q(question,options=None,context=None):
    doc_len = {}
    raw_pred = ""
    # LLM原始答案 这行代码用不到
    cost_time=0.0
    if args.raw_context_pred:
        conte=find_text_by_question_id(f'../data/corpus/raw/{args.dataset}.json',find_question_id(f'../data/eval/{args.dataset}.json',question))
        if conte is None:
            conte='no message!'
            print(conte)
        start_time=time.time()
        raw_pred = search_cache_and_predict(raw_pred, f'{log_path}/raw_pred.json', 'raw_pred', question, model_name, model, tokenizer, lambda: create_prompt(conte,question), maxlen)
        end_time=time.time()
        cost_time=float(end_time-start_time)
    if args.raw_pred:
        start_time=time.time()
        raw_pred = search_cache_and_predict(raw_pred, f'{log_path}/raw_pred.json', 'raw_pred', question, model_name, model, tokenizer, lambda: create_prompt('',question), maxlen)
        end_time=time.time()
        cost_time=float(end_time-start_time)
    # retriever 表示检索到的粗粒度 chunks
    time1=time.time()
    retriever, match_id = vector_search(question)
    # 细粒度的查询 进行排序
    rerank, match_id = sort_section(question, retriever, match_id)
    filter_output = []
    extractor_output = []
    fil_pred = ext_pred = ext_fil_pred = rb_pred = rl_pred = ''

    if args.fil:
        fil_pred = load_cache(f'{log_path}/fil_pred.json', 'fil_pred', question, doc_len, 'Fil')
        if not fil_pred:
            filter_output = filter(question, rerank)
            fil_pred = search_cache_and_predict(fil_pred, f'{log_path}/fil_pred.json', 'fil_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(filter_output), question), maxlen, doc_len, 'Fil')
    
    if args.ext:
        ext_pred = load_cache(f'{log_path}/ext_pred.json', 'ext_pred', question, doc_len, 'Ext')
        if not ext_pred:
            extractor_output = extractor(question, rerank, match_id)
            ext_pred = search_cache_and_predict(ext_pred, f'{log_path}/ext_pred.json', 'ext_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(rerank + extractor_output), question), maxlen, doc_len, 'Ext')

    if args.ext_fil:
        # ext_fil_pred = load_cache(f'{log_path}/ext_fil_pred.json', 'ext_fil_pred', question, doc_len, 'E&F')
        ext_fil_pred=None
        if not ext_fil_pred:
            if not filter_output:
                filter_output = filter(question, rerank)
            if not extractor_output:
                extractor_output = extractor(question, rerank, match_id)
            ext_fil_pred = search_cache_and_predict(ext_fil_pred, f'{log_path}/ext_fil_pred.json', 'ext_fil_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(filter_output + extractor_output), question,options=options,context=context), maxlen, doc_len, 'E&F')
        time2=time.time()
        cost_time=float(time2-time1)    
    if args.rb:
        # rb_pred = load_cache(f'{log_path}/rb_pred.json', 'rb_pred', question, doc_len, 'R&B')
        rb_pred=None
        if not rb_pred:
            rb_pred = search_cache_and_predict(rb_pred, f'{log_path}/rb_pred.json', 'rb_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(rerank), question,options=options,context=context), maxlen, doc_len, 'R&B')
        time2=time.time()
        cost_time=float(time2-time1)
    if args.rl:
        rl_pred = load_cache(f'{log_path}/rl_pred.json', 'rl_pred', question, doc_len, 'R&L')
        if not rl_pred:
            rl_pred = search_cache_and_predict(rl_pred, f'{log_path}/rl_pred.json', 'rl_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(s2l_doc(rerank, match_id, maxlen)[0]), question), maxlen, doc_len, 'R&L')
    
    return question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len,cost_time

def load_cache(cache_path, pred_key, question, doc_len=None, doc_key=None):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['question'] == question:
                    pred_result = data[pred_key]
                    if doc_len is not None and doc_key is not None:
                        doc_len[doc_key] = data["input_len"]
                    return pred_result
    return ''

def search_cache_and_predict(pred_result, cache_path, pred_key, question, model_name, model, tokenizer, create_prompt_func, maxlen, doc_len=None, doc_key=None):
    if not pred_result:
        query = create_prompt_func()
        pred_result, input_len = pred(model_name, model, tokenizer, query, maxlen)
        with open(cache_path, 'a', encoding='utf-8') as f:
            json.dump({'question': question, pred_key: pred_result, "input_len": input_len}, f, ensure_ascii=False)
            f.write('\n')
        if doc_len is not None and doc_key is not None:
            doc_len[doc_key] = input_len
    return pred_result

def s2l_doc(rerank, match_id, maxlen):
    unique_raw_id = []
    contents = []
    s2l_index = {}
    section_index = [id_to_rawid[str(i)] for i in match_id]
    for index, id in enumerate(section_index):
        data = raw_data[id]
        text = data.get("paragraph_text") or data.get("text")
        if id in unique_raw_id and get_word_len(text) < maxlen:
            continue
        if get_word_len(text) >= maxlen:
            content = rerank[index]
        else:
            unique_raw_id.append(id)
            content = text
        s2l_index[len(contents)] = [i for i, v in enumerate(section_index) if v == section_index[index]]
        contents.append(content)
    return contents, s2l_index


def filter(question,rank_docs): 
    
    content="\n".join(rank_docs)
    query=f"{content}\n\nPlease combine the above information and give your thinking process for the following question:{question}."
    think_pro,_=pred(lrag_model_name, lrag_model, lrag_tokenizer, query,lrag_maxlen,1000)
    selected = []

    prompts=[f"""Given an article:{d}\nQuestion: {question}.\nThought process:{think_pro}.\nYour task is to use the thought process provided to decide whether you need to cite the article to answer this question. If you need to cite the article, set the status value to True. If not, set the status value to False. Please output the response in the following json format: {{"status": "{{the value of status}}"}}""" for d in rank_docs]
    pool = ThreadPool(processes=args.MaxClients)
    all_responses=pool.starmap(pred, [(lrag_model_name,lrag_model, lrag_tokenizer,prompt,lrag_maxlen,32) for prompt in prompts])
    for i,r in enumerate(all_responses):
        try:    
            result=json.loads(r[0])
            res=result["status"] 
            if len(all_responses)!=len(rank_docs):
                break     
            if res.lower()=="true":
                selected.append(rank_docs[i])
        except:
            match=re.search("True|true",r[0])
            if match:
                selected.append(rank_docs[i])
    if len(selected)==0:
        selected=rank_docs
    return selected


def r2long_unique(rerank, match_id):
    unique_raw_id = list(set(id_to_rawid[str(i)] for i in match_id))
    section_index = [id_to_rawid[str(i)] for i in match_id]
    contents = [''.join(rerank[i] for i in range(len(section_index)) if section_index[i] == uid) for uid in unique_raw_id]
    return contents, unique_raw_id

def extractor(question, docs, match_id):
    if args.dataset=="medqa":long_docs=docs
    else:long_docs = s2l_doc(docs, match_id, lrag_maxlen)[0]
    content = ''.join(long_docs)
    query = f"{content}.\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{question}"
    response = pred(lrag_model_name, lrag_model, lrag_tokenizer, query, lrag_maxlen, 1000)[0]
    # logger.info(f"cite_passage responses: {all_responses}")
    return [response]


def vector_search(question):
    feature = emb_model.encode([question])
    # 粗粒度查询top_k1个chunks
    distance, match_id = vector.search(feature, args.top_k1)
    # 获取chunks
    content = [chunk_data[int(i)] for i in match_id[0]]
    # 返回chunks和对应的id值 用于映射p
    return content, list(match_id[0])

# 细粒度检索topk2个元素
def sort_section(question, section, match_id):
    q = [question] * len(section)
    if args.dataset=='medqa':
        section= [item["text"] for item in section if "text" in item]
    features = cross_tokenizer(q, section, padding=True, truncation=True, return_tensors="pt").to(device)
    cross_model.eval()
    with torch.no_grad():
        scores = cross_model(**features).logits.squeeze(dim=1)
    sort_scores = torch.argsort(scores, dim=0, descending=True).cpu()
    result = [section[sort_scores[i].item()] for i in range(args.top_k2)]
    match_id = [match_id[sort_scores[i].item()] for i in range(args.top_k2)]
    return result, match_id

def create_prompt(input, question,options=None,context=None):
    # user_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    if args.dataset=='commonsenseqa':
        user_prompt="Assuming you are A know-it-all, I will now ask you some common sense questions and give you 5 options A, B, C, D, E, and additional background on that question. Please think deeply about the questions and background knowledge, and choose the best answer from these options. Please answer this question with a letter option, without adding any words. \nQuestion:"+question+"\nOptions:A."+options['text'][0]+"\nB."+options['text'][1]+"\nC."+options['text'][2]+"\nD."+options['text'][3]+"\nE."+options['text'][4]+"\nbackground Knowledge:"+input
    elif args.dataset=='cosmosqa':
        user_prompt="Assuming you are A know-it-all, I will now ask you some common sense questions based on the context of the conversation, and you will get the answer to the question from that context, and give you possible answers to four options A, B, C, D, and please choose the best answer from those options. In addition, I will provide some additional background knowledge for you to refer to and help with reasoning. Please answer this question with letter options and do not add any words.\nContext:"+context+"\nQuestion:"+question+"\nOptions:A."+options[0]+"\nB."+options[1]+"\nC."+options[2]+"\nD."+options[3]+"\nbackground Knowledge:"+input
    elif args.dataset=='sciq':
        user_prompt="Assuming you are A know-it-all, I will now ask you some  questions and give you 4 options A, B, C, D, E, and additional background on that question. Please think deeply about the questions and background knowledge, and choose the best answer from these options. Please answer this question with a letter option, without adding any words. \nQuestion:"+question+"\nOptions:A."+options[0]+"\nB."+options[1]+"\nC."+options[2]+"\nD."+options[3]+"\nbackground Knowledge:"+input
    elif args.dataset=='medqa':
        user_prompt="Assuming you are A know-it-all, I will now ask you some  questions and give you 5 options A, B, C, D, E, and additional background on that question. Please think deeply about the questions and background knowledge, and choose the best answer from these options. Please answer this question with a letter option, without adding any words. \nQuestion:"+question+"\nOptions:A."+options["A"]+"\nB."+options["B"]+"\nC."+options["C"]+"\nD."+options["D"]+"\nE."+options["E"]+"\nbackground Knowledge:"+input
    elif args.dataset=='bqa':
        user_prompt=f'''
        You are a helpful assistant answering binary (yes/no) questions based on the provided information.
        You will be given:
        1. A question.
        2. A context paragraph.
        3. Retrieved knowledge from external sources.

        Please reason over both the context and the retrieved knowledge, and then answer the question. Your answer must be strictly one word: "yes" or "no", with no other output.
        
        Context: {context}
        Question: {question}
        Retrieved Knowledge: {input}
        Answer:

        '''
    elif args.dataset=='mcqa':
        user_prompt=f'''
        You are a helpful assistant for a multiple-choice question answering task.

        You will be given:
        1. A question.
        2. A context paragraph.
        3. Four answer choices (choice_1 to choice_4).
        4. Retrieved knowledge from external sources.

        Please analyze the question based on the context and the retrieved knowledge. Then select the best answer from the four choices.

        Important: Output only one of the following: "choice_1", "choice_2", "choice_3", or "choice_4". Do not include any punctuation, explanation, or other text.
        
        Context: {context}
        Question: {question}
        Choices:
        choice_1: {options['choice_1']}
        choice_2: {options['choice_2']}
        choice_3: {options['choice_3']}
        choice_4: {options['choice_4']}
        Retrieved Knowledge: {input}
        Answer:

        '''
    else:
        user_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    return user_prompt


if __name__ == '__main__':
    seed_everything(42)
    index_path = f'{args.r_path}/{args.dataset}/vector.index' # Vector index path
    vector = faiss.read_index(index_path)
    if args.dataset!="medqa":
        with open(f'../data/corpus/raw/{args.dataset}.json', encoding='utf-8') as f:
            raw_data = json.load(f)
        with open(f'{args.r_path}/{args.dataset}/id_to_rawid.json', encoding='utf-8') as f:
            id_to_rawid = json.load(f)
    with open(f"{args.r_path}/{args.dataset}/chunks.json", "r") as fin:
        chunk_data = json.load(fin)

    now = datetime.now() 
    now_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/{args.lrag_model or "base"}/{now_time}'
    os.makedirs(log_path, exist_ok=True)

    with open("../config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    model_name = args.model.lower()
    model2path = config["model_path"]
    maxlen = config["model_maxlen"][model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_model = SentenceTransformer(model2path["emb_model"]).to(device)
    cross_tokenizer = AutoTokenizer.from_pretrained(model2path["rerank_model"])
    cross_model = AutoModelForSequenceClassification.from_pretrained(model2path["rerank_model"]).to(device)
    model, tokenizer = load_model_and_tokenizer(model2path, model_name)
    
    if args.lrag_model:
        lrag_model_name = args.lrag_model.lower()
        lrag_maxlen = config["model_maxlen"][lrag_model_name]
        lrag_model, lrag_tokenizer = (model, tokenizer) if model_name == lrag_model_name else load_model_and_tokenizer(model2path, lrag_model_name)
    else:
        lrag_model_name, lrag_model, lrag_tokenizer, lrag_maxlen = (model_name, model, tokenizer, maxlen)
    # set_prompt_tokenizer = AutoTokenizer.from_pretrained(model2path["llama3-8b-instruct-8k"], trust_remote_code=True)
    setup_logger(logger)
    print_args(args)

    questions, answer, raw_preds, rank_preds, ext_preds, fil_preds, longdoc_preds, ext_fil_preds, docs_len = [], [], [], [], [], [], [], [], []
    context=[]
    options=[]
    if args.dataset in "commonsenseqa":
        qs_data = load_dataset("../data/eval/commonsenseqa")['train']
        qs_data = qs_data.select(range(0,200))
        for d in qs_data:
            questions.append(d["question"])
            answer.append(d["answerKey"])
            options.append(d["choices"])
    elif args.dataset in "cosmosqa":
        qs_data = load_dataset('csv', data_files='../data/eval/cosmosqa/train.csv')['train']
        qs_data = qs_data.select(range(0,200))
        label_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        for d in qs_data:
            questions.append(d["question"])
            answer.append(label_mapping[d["label"]])
            temp=[]
            temp.append(d["answer0"])
            temp.append(d["answer1"])
            temp.append(d["answer2"])
            temp.append(d["answer3"])
            options.append(temp)
            context.append(d['context'])
    elif args.dataset in "winograd":
        qs_data = load_dataset("../data/eval/winograd")['test']
        qs_data = qs_data.select(range(0,200))
        for d in qs_data:
            wen=f'''
            prompt = f"""
            You are an expert in natural language understanding. Use the retrieved knowledge and the passage to determine what the pronoun refers to.

            Passage:  
            "{d['text']}"

            Pronoun to resolve:  
            "{d['pronoun']}"

            Candidate referents:  
            - "{d['options'][0]}"  
            - "{d['options'][1]}"

            Retrieved Knowledge:  
            <<Insert retrieved passages or facts here>>

            Question:  
            In the passage above, does the pronoun "{d['pronoun']}" refer to "{d['options'][0]}" or "{d['options'][1]}"?

            Answer (Output only one of the two options exactly as written above, with no explanation):  
            """

            '''
            questions.append(wen)
            answer.append(d["rendered_output"])
    elif args.dataset in "bqa":
        qs_data = load_bqa_ds('../data/eval/bqa.json')
        qs_data = qs_data[:200]
        for d in qs_data:
            questions.append(d["question"])
            answer.append(d["answer"])
            context.append(d['context'])
    elif args.dataset in "mcqa":
        qs_data = load_ds_mcqa('../data/eval/mcqa.json')
        qs_data = qs_data[:200]
        for d in qs_data:
            questions.append(d["question"])
            answer.append(d["answer"])
            options.append(d['choices'])
            context.append(d['context'])
    elif args.dataset in "sciq":
        with open(f'../data/eval/{args.dataset}.json', encoding='utf-8') as f:
            lable_map={"choices_1":"A","choices_2":"B","choices_3":"C","choices_4":"D"}
            qs_data = json.load(f)
            qs_data=qs_data[:200]
            for d in qs_data:
                questions.append(d["question"])
                ch=d["correct_answer"]
                answer.append(lable_map[ch])
                temp=[]
                temp.append(d["choices_1"])
                temp.append(d["choices_2"])
                temp.append(d["choices_3"])
                temp.append(d["choices_4"])
                options.append(temp)
    elif args.dataset in "medqa":
        with open(f'../data/eval/{args.dataset}.json', encoding='utf-8') as f:
            qs_data = json.load(f)
            qs_data=qs_data[:200]
            for d in qs_data:
                questions.append(d["question"])
                answer.append(d["answer_idx"])
                options.append(d["options"])
    else:
        #wikidata squad hotpotqa
        with open(f'../data/eval/{args.dataset}.json', encoding='utf-8') as f:
            qs_data = json.load(f)
            qs_data=qs_data[:200]
        for d in qs_data:
            questions.append(d["question"])
            answer.append(d["answer"] or d["correct_answer"] or d["answer_idx"] or d["correct_answer"])
    
    
    overall_progress = tqdm(total=len(questions), desc="Overall Progress", unit="task")
    out_time=0.0
    for index, query in enumerate(questions):
        if options:op=options[index]
        else:op=None
        if context:con=context[index]
        else:con=None
        question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len,ti= search_q(query,op,con)
        out_time+=ti
        raw_preds.append(raw_pred)
        rank_preds.append(rb_pred)
        ext_preds.append(ext_pred)
        fil_preds.append(fil_pred)
        longdoc_preds.append(rl_pred)
        ext_fil_preds.append(ext_fil_pred)
        if rb_pred:
            print(f'rb_pred:{rb_pred}\n')
            print(f'answer:{answer[index]}\n')
            print(F1_scorer([rb_pred], [answer[index]]))
        if ext_fil_pred:
            print(f'longrag_pred:{ext_fil_pred}\n')
            print(f'answer:{answer[index]}\n')
            print(F1_scorer([ext_fil_pred], [answer[index]]))
        overall_progress.update(1)
        # docs_len.append(doc_len)
    overall_progress.close()
  
    F1 = {
        "raw_pre": F1_scorer(raw_preds, answer),
        "R&B": F1_scorer(rank_preds, answer),
        "Ext": F1_scorer(ext_preds, answer),
        "Fil": F1_scorer(fil_preds, answer),
        "R&L": F1_scorer(longdoc_preds, answer),
        "E&F": F1_scorer(ext_fil_preds, answer),
        "time":out_time
    }
    ext_fil_pred_count=0
    rank_preds_count=0
    if args.ext_fil:
        for p,a in zip(ext_fil_preds,answer):
            if a.lower() in p.lower():ext_fil_pred_count+=1
            else:print(f'wrong! pred: {p} true: {a}\n')
    if args.rb:
        for p,a in zip(rank_preds,answer):
            if a.lower() in p.lower():rank_preds_count+=1
            else:print(f'wrong!  pred: {p} true: {a}\n')
    # eval_result = {"F1": F1, "doc_len": doc_len_eval}
    eval_result = {"F1": F1,"LONGACC":ext_fil_pred_count/200,"RBACC":rank_preds_count/200}
    print(eval_result)
    with open(f"{log_path}/eval_result.json", "w") as fout:
        json.dump(eval_result, fout, ensure_ascii=False, indent=4)