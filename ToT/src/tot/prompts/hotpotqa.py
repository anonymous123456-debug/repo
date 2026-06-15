value_prompt = '''
Evaluate the quality of reasoning steps for long-context-based problems according to the following criteria:
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
Given a passage of text (context) and a question, analyze the possible reasoning steps to extract the correct answer. Expand the reasoning step by step (no more than 5 ideas), ensuring that each step follows a logical progression. Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.  

Follow the format below and don't output anything else.  

### Example:  
**Context:**  
Transatlantic migration refers to the movement of people across the Atlantic Ocean in order to settle on the continents of North and South America. It usually refers to migrations after Christopher Columbus' voyage to the Americas in 1492. ...  
An egg of Columbus or Columbus' egg (Italian: *uovo di Colombo*) refers to a brilliant idea or discovery that seems simple or easy after the fact. The expression refers to an apocryphal story in which Christopher Columbus, having been told that discovering the Americas was inevitable and no great accomplishment, challenges his critics to make an egg stand on its tip. After his challengers give up, Columbus does it himself by tapping the egg on the table to flatten its tip.  

**Question:**  
An egg of Columbus or Columbus' egg refers to a brilliant idea or discovery that seems simple or easy after the fact. The expression refers to an apocryphal story in which 1492, a Spanish-based transatlantic maritime expedition led by Christopher Columbus encountered the Americas, a continent which was previously unknown in Europe, leading to the colonization of the Americas?  

**Current reasoning steps:**  
1. Identify the subject of the question: "An egg of Columbus or Columbus' egg".  
2. Recognize that this phrase refers to a famous story involving Christopher Columbus.  
3. Search the context for references to this term.  
4. The passage defines "Columbus' egg" as an idea that seems obvious in hindsight.  
5. Check for connections between this idea and Columbus' 1492 expedition.  
6. The passage states that Columbus' discovery led to the colonization of the Americas.  
7. Confirm that the phrase "Spanish-based transatlantic maritime expedition led by Christopher Columbus" correctly summarizes this historical event.  

**Next possible ideas:**  
thought1: Ensure that the phrase "Spanish-based transatlantic maritime expedition led by Christopher Columbus" is explicitly mentioned in the passage.  
thought2: Verify that this phrase captures the essence of Columbus’ discovery in 1492.  
thought3: Check if any alternative phrasing exists that may better match the question.  
thought4: If the current phrase is the most accurate answer, finalize it.  
thought5: Cross-check with the broader passage to ensure there are no contradictions.  
thought6: Confirm that no additional inference is required beyond what is stated.  
thought7: Ensure that the reasoning process remains consistent and logical.  

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
Give a question based on a long-context and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Context and Question:
{input}

Reasoning steps:
{reasoning_steps}

By inference, only the final answer is output, without any explanation.
Just write out the final answer, keep it short, and don't organize complicated answers.

The final answer is:
'''
