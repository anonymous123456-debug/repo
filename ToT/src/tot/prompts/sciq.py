value_prompt = '''
Evaluate the quality of reasoning steps for problems in the scientific field according to the following criteria:
1. ** Repeatability ** : Steps that are repeated too many times score lower.
2. ** Support: The reasoning step should be a specific and valid reasoning based on the problem.
3. ** Logical coherence: Reasoning steps should be logically and semantically sound.

The evaluation outputs only three levels: best, good, or bad.
Output only evaluation ratings: best, good, or bad. Contains no other words.)

Question and Options:
{input}

Inference steps to be evaluated:
{current_steps}

Please output the final evaluation based on the above information.
Please output only one word (best or good or bad).
Evaluation (best, good or bad) :
'''


propose_prompt='''
Given a scientific multiple-choice question, analyze the possible reasoning paths and gradually expand the reasoning (no more than 5 ideas). Ensuring that each step follows a logical progression helps determine the most appropriate answer. Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.  

Follow the format below and don't output anything else.  

### Example:  
**Input:**  
Question: Which type of energy is the energy of anything in motion?  

**Options:**  
A: diffuse energy  
B: residual energy  
C: kinetic energy  
D: physiological energy  

**Current reasoning steps:**  
1. The question asks about a type of energy related to motion.  
2. From physics, energy associated with motion is referred to as *kinetic energy*.  
3. The support text explicitly states that *kinetic energy* is the energy of anything in motion.  
4. Reviewing the choices:  
   - *Diffuse energy* (choice A) is not a standard scientific term related to motion.  
   - *Residual energy* (choice B) refers to leftover energy, not motion-related energy.  
   - *Kinetic energy* (choice C) directly matches the definition given in the support text.  
   - *Physiological energy* (choice D) relates to biological processes, not general motion.  
.....  
**Next possible ideas:**  
thought1: *Kinetic energy* (choice C) is the correct answer because it aligns with the definition in the support text.  
thought2: *Diffuse energy* (choice A) is incorrect because it does not describe a standard energy type in physics.  
thought3: *Residual energy* (choice B) is incorrect because it refers to leftover energy, not motion-related energy.  
thought4: *Physiological energy* (choice D) is incorrect as it is related to biological processes rather than motion.  
thought5: Understanding energy transformations can help reinforce why kinetic energy is associated with moving objects.  
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
Give a multiple choice question related to a scientific field and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Questions and Options:
{input}

Reasoning steps:
{reasoning_steps}

By inference, output only the corresponding option letters (e.g., A, B, C, D) without any explanation.
You only need to output one uppercase letter, you only need to output the uppercase letter of the answer option.
The final answer is:
'''