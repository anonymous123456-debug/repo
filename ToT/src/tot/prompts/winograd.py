# Standard prompt


standard_prompt = '''Determine which option the given pronoun refers to.

Sentence: The city councilmen refused the demonstrators a permit because they feared violence.
Pronoun: they
Options: the city councilmen, the demonstrators
Answer: the city councilmen

Sentence: Joan made sure to thank Susan for all the help she had given.
Pronoun: she
Options: Joan, Susan
Answer: Susan

Sentence: {input}
Answer:
'''

# Chain-of-Thought (CoT) prompt
cot_prompt = '''Determine which option the given pronoun refers to by reasoning through the sentence step by step.

Sentence: The trophy does not fit into the brown suitcase because it is too large.
Steps:
- "It" refers to the noun that has a size attribute.
- The trophy has a size attribute (large), while the suitcase has a size attribute (small).
- "It" is described as "too large," which matches the trophy.
Answer: the trophy

Sentence: Joan made sure to thank Susan for all the help she had given.
Steps:
- "She" refers to the person who had given help.
- The sentence suggests that Susan gave the help.
- Thus, "she" refers to Susan.
Answer: Susan

Sentence: {input}
Steps:
'''


value_prompt='''
Evaluate the given reasoning steps based on the following criteria:  
1. **Repetitiveness**: Steps with excessive repetition receive a lower score.  
2. **Supportiveness**: Steps should effectively contribute to identifying the pronoun referent.  
3. **Logical Coherence**: The reasoning steps should be logically sound.  
The evaluation only output the 3 levels:best, good, or bad.
Output only evaluation ratings: best, good, or bad. Contains no other words.)

{input}  
Current reasoning steps:  
{current_steps}  

Please output the final evaluation based on the above information. Output only evaluation ratings: best, good, or bad. Contains no other words.)
Output only one word(best, good or bad)
Evaluation (best, good, or bad):  
'''

propose_prompt = '''Given a sentence with an ambiguous pronoun, analyze possible referents and extend the reasoning step by step(No more than 5 thoughts).

{input}

Current reasoning steps:
{current_steps}

Next possible thought(s):
thought1:...
thought2:...
.......
'''

last_prompt = '''Given a sentence with an ambiguous pronoun, analyze possible referents and make the final decision(only output 0 or 1):

   {input}
   
   Reasoning steps:{reasoning_steps}
   
   Based on the reasoning, output only the corresponding number (0 or 1) without explanation.  
    
   Final answer:'''
