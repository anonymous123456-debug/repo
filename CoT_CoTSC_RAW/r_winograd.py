from datasets import load_dataset
import logging
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载数据集并选择前 200 个样本
ds = load_dataset("./data/winograd")['test']
ds = ds.select(range(0, 200))

# 加载本地 LLaMA 模型
modelname = '../llama3-8b'
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# 保存生成文本
rag_texts = []

# 主循环
overall_progress = tqdm(total=len(ds), desc="Generating RAG Corpus", unit="sample")
for idx, sample in enumerate(ds):
    passage = sample['text']
    pronoun = sample['pronoun']
    opt0 = sample['options'][0]
    opt1 = sample['options'][1]

    # 构造 Prompt
    prompt = (
        f"You are given a short passage for pronoun resolution.\n"
        f"Passage: \"{passage}\"\n"
        f"The pronoun \"{pronoun}\" appears in the passage and may refer to either \"{opt0}\" or \"{opt1}\".\n"
        f"Please provide background knowledge, commonsense facts, or relevant world knowledge that can help determine the correct referent of the pronoun in this specific context. "
        f"The result should be a fluent paragraph without listing or bullet points."
    )

    # 构造 Chat 格式
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to('cuda')

    # 模型生成
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.get('attention_mask'),
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    # 移除 prompt 部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 存储结果
    rag_texts.append({
        "id": idx,
        "content": passage,
        "pronoun": pronoun,
        "option_0": opt0,
        "option_1": opt1,
        "text": response.strip()
    })

    print(f"\n--- Sample {idx} ---\n{response}\n")
    overall_progress.update(1)

# 保存为 JSON 文件
with open("./data/corpus/raw/winograd.json", "w", encoding="utf-8") as f:
    json.dump(rag_texts, f, indent=4, ensure_ascii=False)

overall_progress.close()
