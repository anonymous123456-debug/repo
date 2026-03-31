from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,GPT2Tokenizer, GPT2Model
import torch

# 加载本地模型
# modelname='../../../servy/llama3-8b'
# modelname='/share/home/ncu_418000240001/servy/llama3-8b'
# modelname='/share/home/ncu_418000240001/qwen2.5-7b'
# modelname='/share/home/ncu_418000240001/qwen3-8b'
# modelname='/share/home/ncu_418000240001/qwen3-8b'
# modelname='/share/home/ncu_418000240001/deepseekR1'
# modelname='/share/home/ncu_418000240001/phi-3.8b'
modelname='/share/home/ncu_418000240001/smolLM2-1.7b'
# modelname='/share/home/ncu_418000240001/r1_llama38b'
# modelname='/share/home/ncu_418000240001/mistral'
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
mod = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')


# def llama3_8b(prompt):
#     messages = [{"role": "user", "content": prompt}]
#     text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
#     model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
#     generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=1000,pad_token_id=tokenizer.eos_token_id)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response
total_prompt_tokens=total_completion_tokens=0
def gpt(prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=6000, n=1, stop=None):
    global total_prompt_tokens,total_completion_tokens
    """
    使用 Llama 模型生成多个结果。

    :param prompt: 输入的提示文本
    :param n: 生成结果的数量
    :param stop: 停止生成的标记（可选）
    :return: 一个包含多个生成结果的列表
    """
    messages = [{"role": "user", "content": prompt}]

    generated_responses = []
    print(f'prompt:{prompt}\n')
    for _ in range(n):  # 根据 n 生成多个结果
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
         # 统计 prompt token 数
        prompt_tokens = len(model_inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens
        generated_ids = mod.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=1000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
         # 统计 completion token 数
        completion_tokens = len(generated_ids[0])
        total_completion_tokens += completion_tokens
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_responses.append(response)
    return generated_responses


    
def gpt_usage(backend="gpt-4"):
    global total_completion_tokens,total_prompt_tokens
    print(total_completion_tokens)
    print(total_prompt_tokens)
    print(total_prompt_tokens+total_completion_tokens)