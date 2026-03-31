from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import json

def load_ds(filepath):
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
# 假设你有BQA数据集，字段为 context 和 question
ds = load_ds("./data/logicbench/logic_bqa.json")
ds= ds[:200] # 可调整样本数量

modelname = '../llama3-8b'
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

results = []

progress = tqdm(total=len(ds), desc="Generating RAG for BQA")

for idx, q in enumerate(ds):
    context = q['context']
    question = q['question']

    prompt = f"""
You are provided with a Boolean question and its prerequisite context.

Prerequisite:
{q['context']}

Question:
{q['question']}

Your task is to generate detailed background knowledge, external commonsense, and relevant real-world information that can help determine whether the answer to the question is True or False given the prerequisite.

Include as much relevant information as possible, such as definitions, cause-effect relationships, typical scenarios, examples, or related facts that a person would retrieve or recall when answering such a question. The output should be a rich, coherent paragraph with no blank lines or bullet points.
"""

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

    outputs = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=4000,
        pad_token_id=tokenizer.eos_token_id
    )
    # 移除 prompt
    outputs = [o[len(i):] for o, i in zip(outputs, model_inputs.input_ids)]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    results.append({
        "id": idx,
        "context": context,
        "question": question,
        "text": response.strip()
    })

    print(f"\n--- {idx} ---\n{response}\n")
    progress.update(1)

progress.close()

# 保存为JSON
with open("bqa_rag_corpus.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
