value_prompt = '''
Evaluate the quality of reasoning steps for problems in the medical field according to the following criteria:
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
Given a clinical case scenario with a multiple-choice question, analyze the possible diagnostic and treatment reasoning paths. Expand the reasoning step by step (no more than 5 ideas), ensuring that each step follows a logical progression. Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.  

Follow the format below and don't output anything else.  

### Example:  
**Input:**  
Question: A 55-year-old patient is brought to the emergency department because he has had sharp chest pain for the past 3 hours. He reports that he can only take shallow breaths because deep inspiration worsens the pain. He also reports that the pain increases with coughing. Two weeks ago, he underwent cardiac catheterization for an acute myocardial infarction. Current medications include aspirin, ticagrelor, atorvastatin, metoprolol, and lisinopril. His temperature is 38.5°C (101.1°F), pulse is 55/min, respirations are 23/min, and blood pressure is 125/75 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard when the patient is sitting upright and during expiration. An ECG shows diffuse ST elevations and ST depression in aVR and V1. An echocardiography shows no abnormalities. Which of the following is the most appropriate treatment in this patient?  

**Options:**  
A: Start heparin infusion  
B: Administer nitroglycerin  
C: Increase aspirin dose  
D: Perform pericardiocentesis  
E: Perform CT angiography  

**Current reasoning steps:**  
1. The patient presents with pleuritic chest pain that worsens with deep inspiration and coughing, which is characteristic of pericarditis.  
2. The recent history of myocardial infarction and current symptoms suggest post-cardiac injury syndrome (Dressler syndrome).  
3. The presence of a high-pitched scratching sound on auscultation is consistent with a pericardial friction rub, further supporting the diagnosis.  
4. The ECG findings of diffuse ST elevations with ST depression in aVR and V1 are classic for pericarditis.  
5. The absence of echocardiographic abnormalities suggests that there is no significant pericardial effusion requiring drainage.  
.....  
**Next possible ideas:**  
thought1: The most appropriate treatment for Dressler syndrome is anti-inflammatory therapy, typically with aspirin or NSAIDs.  
thought2: *Start heparin infusion* (choice A) is incorrect because this is not an acute coronary syndrome or pulmonary embolism.  
thought3: *Administer nitroglycerin* (choice B) is not appropriate because the chest pain is not ischemic in origin.  
thought4: *Increase aspirin dose* (choice C) is appropriate because aspirin is the first-line treatment for pericarditis after myocardial infarction.  
thought5: *Perform pericardiocentesis* (choice D) is not needed because there is no significant effusion on echocardiography.  
thought6: *Perform CT angiography* (choice E) is unnecessary, as the diagnosis is clinically evident based on history and ECG findings.  
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
Give a single choice question related to medicine and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Questions and Options:
{input}

Reasoning steps:
{reasoning_steps}

By inference, output only the corresponding option letters (e.g., A, B, C, D, E) without any explanation.
You only need to output one uppercase letter, you only need to output the uppercase letter of the answer option.
The final answer:
'''

# last_prompt="""
# Give a single choice question related to medicine and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

# Questions and Options:
# {input}

# Reasoning steps:
# {reasoning_steps}

# By inference, output only the corresponding option letters (e.g., A, B, C, D, E) without any explanation.
# You only need to output one uppercase letter, you only need to output the uppercase letter of the answer option.
# Finally, output only the single uppercase letter of the final answer (for example: A). Do not output any other characters, words, or punctuation.
# """