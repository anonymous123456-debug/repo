      
import json
import backoff
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,GPT2Tokenizer, GPT2Model
import torch
# @backoff.on_exception(backoff.expo, (Exception), max_time=500)
# def call_api(prompt,model,max_new_tokens):
#     if "glm" in model:
#         res=glm(prompt,model, max_new_tokens)
#     elif "gpt" in model:
#         res=gpt(prompt,model,max_new_tokens)
#         if not res:
#             prompt=remove_consecutive_repeated_sentences(prompt)
#             res=gpt(prompt,model,max_new_tokens)
#     assert res != None
#     return res

# 作者是吊api这里我们 部署自己的大模型进行返回。
tokenizer = AutoTokenizer.from_pretrained("../chatglm3-6b-32k", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../chatglm3-6b-32k", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
# tokenizer=None
# model=None
def glm(prompt,model,max_tokens):
        history=[]
        tokenizer = AutoTokenizer.from_pretrained("../chatglm3-6b-32k", trust_remote_code=True)
        mo = AutoModelForCausalLM.from_pretrained("../chatglm3-6b-32k", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        response,history= mo.chat(tokenizer, prompt,history=history,max_new_tokens=2000, temperature=0.99, num_beams=1, do_sample=False)
        print(f'response:{response}\n')
        tokenizer=None
        mo=None
        return response
@backoff.on_exception(backoff.expo, (Exception), max_time=500)
def call_api(prompt,model,max_new_tokens):
    if "glm" in model:
        res=glm(prompt,model, max_new_tokens)
    assert res != None
    return res

if __name__ == "__main__":

    print(call_api("Hello","gpt-3.5-turbo",100))
    print(call_api("Hello","gpt-3.5-turbo-16k",100))
    print(call_api("Hello","glm-4",100))
    print(call_api("Hello","chatglm_turbo",100))
    



    