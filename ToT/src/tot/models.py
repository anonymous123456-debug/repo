import os

from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,GPT2Tokenizer, GPT2Model
import torch

# Load the local model path or public Hugging Face model ID from the environment.
modelname = os.getenv("TOT_MODEL_PATH", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
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
    Generate one or more responses with the configured local model.

    :param prompt: Input prompt text.
    :param n: Number of responses to generate.
    :param stop: Optional stop marker.
    :return: A list of generated responses.
    """
    messages = [{"role": "user", "content": prompt}]

    generated_responses = []
    print(f'prompt:{prompt}\n')
    model_device = next(mod.parameters()).device
    for _ in range(n):  # Generate n independent responses.
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model_device)
        # Count prompt tokens.
        prompt_tokens = len(model_inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens
        generated_ids = mod.generate(model_inputs.input_ids,attention_mask=model_inputs.get('attention_mask'), max_new_tokens=1000,pad_token_id=tokenizer.eos_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # Count completion tokens.
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
