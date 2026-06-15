value_prompt = '''
Evaluate the quality of reasoning steps for context-based problems according to the following criteria:
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
Given a passage of text (context) and a question, analyze the possible reasoning steps to extract the correct answer.  Expand the reasoning step by step (no more than 5 ideas), ensuring that each step follows a logical progression.  Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.

Follow the format below and don't output anything else.

### Example:
**Context:**
Anthropologists, along with other social scientists, are working with the US military as part of the US Army's strategy in Afghanistan.  The Christian Science Monitor reports that "Counterinsurgency efforts focus on better grasping and meeting local needs" in Afghanistan, under the Human Terrain System (HTS) program;  in addition, HTS teams are working with the US military in Iraq.  In 2009, the American Anthropological Association's Commission on the Engagement of Anthropology with the US Security and Intelligence Communities released its final report concluding, in part, that, "When ethnographic investigation is determined by military missions, not subject to external review, where data collection occurs in the context of war, integrated into the goals of counterinsurgency,  and in a potentially coercive environment – all characteristic factors of the HTS concept and its application – it can no longer be considered a legitimate professional exercise of anthropology.  In summary, while we stress that constructive engagement between anthropology and the military is possible, CEAUSSIC suggests that the AAA emphasize the incompatibility of HTS with disciplinary ethics and practice for job seekers and that it further recognize the problem of allowing HTS to define the meaning of 'anthropology' within DoD."

**Question:**
Who are anthropologists working with along with other social scientists?

**Current reasoning steps:**
1.  Identify the key entity in the question: "anthropologists."
2.  Look for the phrase "working with" to determine their collaborators.
3.  The phrase "Anthropologists, along with other social scientists, are working with the US military" directly answers the question.
4.  Verify that the answer is explicitly stated in the passage.
5.  Ensure that "US military" is the most specific and relevant answer.
.....
**Next possible ideas:**
thought1: The key phrase "Anthropologists, along with other social scientists, are working with the US military" confirms the answer.
thought2: No additional inference is needed since the answer is explicitly mentioned.
thought3: The phrase "working with" ensures that the US military is the correct entity.
thought4: Other details in the passage do not change or contradict this interpretation.
thought5: The final answer is "US military."
.....
### Now process the following input:

**Context and Question:**
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
Give a question based on a context and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Context and Question:
{input}

Reasoning steps:
{reasoning_steps}

By inference, only the final answer is output, without any explanation.
Just write out the final answer, keep it short, and don't organize complicated answers.

The final answer is:
'''
