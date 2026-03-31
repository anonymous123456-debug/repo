from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import json

def load_ds_mcqa(filepath):
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        result.append({
            'id': d['id'],
            'choices': d['choices'],  # 字典结构
            'context': d['context'],
            'answer': d['answer'],
            'question': d['question']
        })
    return result

# 加载数据
ds = load_ds_mcqa("./data/logicbench/logicbench_mcqa.json")
ds = ds[:200]

# 加载模型
modelname = '../llama3-8b'
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    modelname, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map='auto'
)

results = []
progress = tqdm(total=len(ds), desc="Generating MCQA RAG")

for idx, q in enumerate(ds):
    context = q['context'].strip()
    question = q['question'].strip()
    choices = q['choices']  # 是 dict

    # 构造包含 Prerequisites 的 RAG prompt
    prompt = f"""
You are given a multiple-choice question and some background context (Prerequisites). Your task is to generate detailed background knowledge and reasoning paths to help someone choose the correct answer.

Use the context and connect it to the question and each of the options. Be as detailed as possible, including definitions, examples, and causal relationships. Output a single paragraph without bullet points or newlines.
Include as much relevant information as possible, such as definitions, cause-effect relationships, typical scenarios, examples, or related facts that a person would retrieve or recall when answering such a question.  The output should be a rich, coherent paragraph with no blank lines or bullet points.
Prerequisites:
{context}

Question:
{question}

Options:
A. {choices['choice_1']}
B. {choices['choice_2']}
C. {choices['choice_3']}
D. {choices['choice_4']}
"""

    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to("cuda")

    outputs = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    outputs = [o[len(i):] for o, i in zip(outputs, model_inputs.input_ids)]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    results.append({
        "id": q['id'],
        "question": question,
        "context": context,
        "choices": choices,
        "answer": q['answer'],
        "text": response.strip()
    })

    print(f"\n--- {idx} ---\n{response}\n")
    progress.update(1)

progress.close()

with open("mcqa_rag_corpus.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
