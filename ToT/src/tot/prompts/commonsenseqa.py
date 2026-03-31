value_prompt = '''
Evaluate the quality of the reasoning steps on common sense problems according to the following criteria:
1. ** Repeatability ** : Steps that are repeated too many times score lower.
2. ** Support: The reasoning step should be a specific and valid reasoning based on the problem.
3. ** Logical coherence: Reasoning steps should be logically and semantically sound.

The evaluation outputs only three levels: best, good, or bad.
Output only evaluation ratings: best, good, or bad. Contains no other words.)

Context and Question:
{input}

Inference steps to be evaluated:
{current_steps}

Please output the final evaluation based on the above information.
Please output only one word (best or good or bad).
Evaluation (best, good or bad) :
'''


propose_prompt='''
Given a multiple-choice question, analyze the possible reasoning paths and gradually expand the reasoning (no more than 5 ideas). Ensuring that each step follows a logical progression helps determine the most appropriate answer. Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.

Follow the format below and don't output anything else.

### Example:
**Input:**  
Questions: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?  

**Options:**  
A:ignore  
B:enforce  
C:authoritarian  
D:yell at  
E:avoid  

**Current reasoning steps:**  
1. The phrase *"punishing blow"* implies that the sanctions had a strong negative impact on the school.  
2. The question asks about how the sanctions related to the school's efforts to change.  
3. If the sanctions *supported* the school's efforts, they would have reinforced or enforced them.  
4. If the sanctions *opposed* the school's efforts, they would have negated or ignored them.  
5. The word *"seemed to"* suggests an effect that can be perceived rather than explicitly stated.  
.....
**Next possible ideas:**  
thought1: The word *ignore* (choice A) suggests that the sanctions disregarded the school’s changes, which aligns with the idea that they did not acknowledge improvements.  
thought2: *Enforce* (choice B) does not fit well because sanctions are usually punitive rather than supportive.  
thought3: *Authoritarian* (choice C) describes a strict governing style rather than a response to change, making it less relevant.  
thought4: *Yell at* (choice D) is informal and does not fit the context of institutional sanctions.  
thought5: *Avoid* (choice E) implies evasion, which does not match the idea of a punishing blow.  
.....
### Now process the following input:  

**Input:**  
{input}  

**Current reasoning steps:**  
{current_steps}  

**Next possible ideas:**  
thought1: ...  
thought2: ...  
thought3: ...  
...
'''

last_prompt = '''
Give a single choice question related to common sense, carefully analyze the reasoning process, and determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Question and Options: 

{input}

Reasoning steps: 

{reasoning_steps}

By inference, output only the corresponding option letters (e.g., A, B, C, D, E) without any explanation.

You only need to output one uppercase letter, you only need to output the uppercase letter of the answer option.

Final Answer: 
'''