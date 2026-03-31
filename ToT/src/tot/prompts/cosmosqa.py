value_prompt = '''
Evaluate the quality of the context-based common-sense problem reasoning steps against the following criteria:
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
Give a single choice question based on context, analyze possible reasoning paths, and develop reasoning step by step (no more than 5 ideas). Ensuring that each step follows a logical progression helps determine the most appropriate answer. Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.

Follow the format below and don't output anything else.

### Example:

**Input:**  
Context: Good Old War and Person L: I saw both of these bands Wednesday night, and they both blew me away. Seriously. Good Old War is acoustic and makes me smile. I really can not help but be happy when I listen to them; I think it's the fact that they seemed so happy themselves when they played.

Question: In the future, will this person go to see other bands play?

**Options:**  
A: None of the above choices.  
B: This person likes music and likes to see the show, they will see other bands play.  
C: This person only likes Good Old War and Person L, no other bands.  
D: Other bands are not on tour and this person cannot see them.

**Current reasoning steps:**  
1. The person expresses a strong positive reaction to the bands they saw.  
2. They describe Good Old War as making them happy, which suggests they enjoy live music experiences.  
3. The phrase "I really cannot help but be happy when I listen to them" implies that they find joy in live performances.  
4. Enjoying these bands does not necessarily mean they dislike all other bands.  
......

**Next possible thoughts:**  
thought1: Option A (None of the above) is too vague and does not align with the given context.  
thought2: Option B suggests that enjoyment of live music could lead them to see other bands, which aligns with the reasoning.  
thought3: Option C is too restrictive, as there is no evidence that they only like these two bands.  
thought4: Option D assumes external constraints (tour availability), which is not supported by the given context.  
......

### Now process the following input:  
**Input:**  
{input}

**Current reasoning steps:**  
{current_steps}

**Next possible thoughts:**  
thought1:...  
thought2:...  
thought3:...  
...
'''

last_prompt = '''
Give a common sense multiple choice question based on context and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Questions and Options:
{input}

Reasoning steps:
{reasoning_steps}

By inference, output only the corresponding option letters (e.g., A, B, C, D, E) without any explanation.
You only need to output one uppercase letter, you only need to output the uppercase letter of the answer option.

The final answer:
'''

