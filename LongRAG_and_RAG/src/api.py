import json
import backoff
import os
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

CHATGLM_MODEL_PATH = os.environ.get("CHATGLM_MODEL_PATH", "../chatglm3-6b-32k")
_tokenizer = None
_model = None


def load_glm_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(CHATGLM_MODEL_PATH, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            CHATGLM_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
    return _tokenizer, _model


def glm(prompt,model,max_tokens):
    history=[]
    tokenizer, mo = load_glm_model()
    response,history= mo.chat(tokenizer, prompt,history=history,max_new_tokens=max_tokens, temperature=0.99, num_beams=1, do_sample=False)
    print(f'response:{response}\n')
    return response


@backoff.on_exception(backoff.expo, (Exception), max_time=500)
def call_api(prompt,model,max_new_tokens):
    res = None
    if "glm" in model:
        res=glm(prompt,model, max_new_tokens)
    assert res != None
    return res

if __name__ == "__main__":

    print(call_api("Hello","gpt-3.5-turbo",100))
    print(call_api("Hello","gpt-3.5-turbo-16k",100))
    print(call_api("Hello","glm-4",100))
    print(call_api("Hello","chatglm_turbo",100))
    



    
