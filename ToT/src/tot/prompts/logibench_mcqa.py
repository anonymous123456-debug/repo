value_prompt = '''
Evaluate the quality of the reasoning steps for a contextual problem according to the following criteria:
1. ** Repeatability ** : Steps that are repeated too many times score lower.
2. ** Support ** : Reasoning steps should be specific and valid reasoning based on context and problem.
3. ** Logical coherence: Reasoning steps should be logically and semantically sound.

The evaluation outputs only three levels: best, good, or bad.
Output only evaluation ratings: best, good, or bad. Contains no other words.)

Context and Question:
{input}

Inference steps to be evaluated:
{current_steps}

Please output the final evaluation based on the above information. Output only evaluation ratings: best, good, or bad. Contains no other words.)
Output only one word(best, good or bad)
Evaluation (best, good or bad) :
'''


propose_prompt='''
Given a multi-choice problem, analyze possible reasoning paths and gradually expand the reasoning (no more than 5 ideas).  Ensuring that each step follows a logical progression helps determine the most suitable conclusion.  Your task is to generate no more than 5 possible ideas for the next step based on the current reasoning step.

Follow the exact format below and do not output anything else.

### Example:
**Input:**
Context: If an individual consumes a significant amount of water, they will experience a state of hydration.  Conversely, if excessive amounts of sugar are ingested, a sugar crash will ensue.  It is known that at least one of the following statements is true: either Jane consumes ample water or she will not experience a sugar crash.  However, the actual veracity of either statement remains ambiguous, as it could be the case that only the first statement is true, only the second statement is true, or both statements are true.
Question: Based on the context, what conclusion would be deemed most suitable?
Choices:
- (choice_1) If Jane consumes ample water, she will experience a sugar crash.
- (choice_2) John will feel hydrated or he won't experience a sugar crash.
- (choice_3) She will feel hydrated or she doesn't eat too much sugar.
- (choice_4) Jane won't feel hydrated or she will eat too much sugar.

**Current reasoning steps:**
1.  If Jane consumes ample water, she will experience hydration.
2.  If Jane does not ingest excessive sugar, she will not experience a sugar crash.
3.  The problem states that at least one of these conditions must hold, but it does not specify which one.
4.  We need to determine which choice aligns most logically with the given statements.

**Next possible thought:**
thought1: If Jane consumes ample water, then hydration is guaranteed, contradicting choice (choice_4).
thought2: If Jane avoids excessive sugar intake, then she will not experience a sugar crash, aligning with choice (choice_3).
thought3: The phrase "at least one must be true" suggests that negating both statements simultaneously is incorrect, weakening choice (choice_4).
thought4: Choice (choice_1) contradicts the given premises, as water consumption does not directly lead to a sugar crash.
thought5: Since "or" statements allow for flexibility, choice (choice_2) and choice (choice_3) remain plausible.
...

### Now process the following input:
**Input:**
{input}

**Current reasoning steps:**
{current_steps}

**Next possible thought:**
thought1: ...
thought2: ...
thought3: ...
...
'''

last_prompt = '''
Give a multiple choice question related to logical reasoning, carefully analyze the reasoning process, and determine the most appropriate choice. Your task is to evaluate the inference steps provided and output the corresponding selection identifiers (for example, choice_1,choice_2,choice_3,choice_4).

Input :* * * *
{input}

* * Reasoning steps :* *
{reasoning_steps}

By inference, only the corresponding selection identifier (such as choice_1,choice_2,choice_3,choice_4) is output without any explanation.

* * Final Answer :* *
'''