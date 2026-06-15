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
Mark Kenneth Woods (date of birth unknown) is a Canadian comedy writer, actor, producer, director, and TV host. Theodred II was a medieval Bishop of Elmham. The date of Theodred's consecration is unknown, but the date of his death was sometime between 995 and 997. Etan Boritzer (born 1950) is an American writer of children's literature who is best known for his book *"What is God?"* first published in 1989. ...  
Evangelista Andreoli (27 June 1810 – 16 June 1875) was an Italian organist, pianist, and teacher. Born in (now in the *"comune"* Cavezzo), Modena, he moved to nearby Mirandola, where he played organ and taught at the music school for his entire life. ...  
Carlo Andreoli (January 8, 1840 – January 22, 1908) was an Italian pianist. He was born in Mirandola, Modena to the musical family of Evangelista Andreoli; his brothers included Guglielmo the Elder and the Younger. ...  

**Question:**  
When is Carlo Andreoli's father's birthday?  

**Current reasoning steps:**  
1. Identify the subject of the question: "Carlo Andreoli's father."  
2. Search the context for references to Carlo Andreoli.  
3. Carlo Andreoli's father is Evangelista Andreoli.  
4. Locate Evangelista Andreoli's birthdate in the passage.  
5. The passage states that Evangelista Andreoli was born on *27 June 1810*.  
6. Ensure that no contradictory information exists in the passage.  
7. Confirm that the extracted birthdate is clearly associated with Evangelista Andreoli.  

**Next possible ideas:**  
thought1: The key phrase "Evangelista Andreoli (27 June 1810 – 16 June 1875)" confirms the answer.  
thought2: Since Evangelista Andreoli is explicitly mentioned as Carlo Andreoli’s father, this is the correct birthdate.  
thought3: No additional inference is needed since the birthdate is directly stated.  
thought4: The final answer is *27 June 1810*.  
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
Give a question based on a long-context and carefully analyze the reasoning process to determine the most appropriate answer. Your task is to evaluate the reasoning steps provided and output the corresponding answer options.

Context and Question:
{input}

Reasoning steps:
{reasoning_steps}

By inference, only the final answer is output, without any explanation.
Just write out the final answer, keep it short, and don't organize complicated answers.

The final answer is:
'''
