value_prompt = '''
Evaluate the given reasoning steps based on the following criteria:  
1. **Repetitiveness**: Steps with excessive repetition receive a lower score.  
2. **Supportiveness**: The steps should effectively help to understand the context of use and make reasonable thinking about it.
3. **Logical Coherence**: Reasoning steps should be logically and semantically sound.
The evaluation only output the 3 levels:best, good, or bad.

{input}
Current reasoning steps:  
{current_steps}  

Please output the final evaluation based on the above information. Output only evaluation ratings: best, good, or bad. Contains no other words.)
Evaluation (best, good, or bad):  
'''

propose_prompt = '''
Given a binary problem, analyze possible reasoning paths and gradually expand the reasoning (no more than 5 ideas). Making sure that each step is logical helps determine the final answer: "Yes" or "no." Your task is to come up with no more than 5 possible ideas for the next step based on the current reasoning step.

Follow the exact format below and do not output anything else.
### Example:
**Input:**
Context: If an individual consumes a significant amount of water, they will experience a state of hydration.  Conversely, if excessive amounts of sugar are ingested, a sugar crash will ensue.  It is known that at least one of the following statements is true: either Jane consumes ample water or she will not experience a sugar crash.  However, the actual veracity of either statement remains ambiguous, as it could be the case that only the first statement is true, only the second statement is true, or both statements are true.
Question: Can we say at least one of the following must always be true?  (a) she will feel hydrated and (b) she doesn't eat too much sugar

**Current reasoning steps:**
1.  If Jane consumes ample water, she will experience hydration.
2.  If Jane does not ingest excessive sugar, she will not experience a sugar crash.
3.  The problem states that at least one of these statements is true, but it does not specify which one.

**Next possible thought:**
thought1: If Jane consumes ample water, statement (a) must be true.
thought2: If Jane avoids excessive sugar intake, statement (b) must be true.
thought3: Since at least one of these conditions is guaranteed, at least one of (a) or (b) must be true.
....

### Now process the following input:  
**Input:**  
{input}

**Current reasoning steps:**  
{current_steps}

**Next possible thought:** 
thought1:...
thought2:...
thought3:...
...

'''

last_prompt = '''
Given a binary question, analyze the reasoning steps and determine the final answer (only output "yes" or "no").

{input}

Reasoning steps:
{reasoning_steps}

By inference, answer the question only 'yes' or 'no' without any explanation.

Final answer:
'''