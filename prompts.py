

###############  START OF PROMPTS USED IN PAPER #################################################################################################################################################################################################################################

standard_prompt = '''Write a short and coherent story about {topic} that incorporates the answers to the following {n} questions: {questions}
'''

## prompts for self-refinement ##
self_refine_feedback_prompt = '''{question_answer}
---
Reflect on the response. Analyze the correctness of the information provided, and the coherence of the story. Provide critque to help improve the response. Your feedback:
'''

self_refine_refinement_prompt = '''{question_answer}
---
Feedback: {feedback}
---
Based on your initial response and the subsequent feedback, revise the response. Your revised response:
'''
#################################


cot_prompt = '''Write a short and coherent story about {topic} that incorporates the answers to the following {n} questions: {questions}

Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Story:
Your story here.
'''

spp_prompt = '''When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Write a short and coherent story about {topic} that incorporates the answers to the following {n} questions: {questions}
'''


###############  END OF PROMPTS USED IN PAPER ##########################################################################################################################################################################################################



####################### ANSWERING AGENT PROMPTS ##########################
answer_all_prompt = '''
Answer the following {n} questions: {questions}

Answers:
Answers here.
'''
######################## END OF ANSWERING AGENT PROMPTS ##########################




###################### CHECKING AGENT PROMPTS ##########################

double_check_one_at_a_time_prompt = '''
Question: {question}

Proposed answer: {proposed_answer}

Please double-check the proposed answer for factual correctness, and write a revised answer if necessary. If the proposed answer is correct, simply repeat it. If it is incorrect, provide the correct answer.

Revised answer:
Revised answer here.
'''

double_check_all_prompt = '''
Questions: {questions}

Proposed answers: {proposed_answers}

Please double-check the proposed answers for factual correctness, and write a revised answer if necessary. If a proposed answer is correct, simply repeat it. If it is incorrect, provide the correct answer.

Revised answers:
Revised answers here.
'''


confidence_assessment_question_prompt = '''
Question: {question}

Answer:
Your answer here
'''

confidence_assessment_assess_prompt = '''
Question: {question}

Proposed answer: {proposed_answer}

Is the proposed answer:
    (A) True
    (B) False
The proposed answer is: 
'''

confidence_assessment_write_prompt = ''' 
Write a short and coherent story about {topic} that incorporates the following {n} points: {answers}
'''

####################### END OF CHECKING AGENT PROMPTS ##########################


# questions_first_prompt = '''Answer the following {n} questions first: {questions}

# Now incorporate them into a short and coherent story about {topic}.

# Answers:
# Answers to the questions here.

# Story:
# Story here.
# '''

# persona_prompt = '''You are an expert storyteller and a master of trivia. Your task is to write a short, coherent story about {topic}. You must skillfully and accurately weave the answers to the following {n} questions into your story: {questions}.

# Make sure the story is engaging and that all the trivia answers are factually correct.
# '''

# structured_decomposition_prompt = '''Follow these steps precisely:
# Step 1: First, list the correct answers to the following {n} questions.
# Questions: {questions}

# Step 2: Next, create a brief, 3-point outline for a story about {topic} that logically incorporates all the answers from Step 1.

# Step 3: Finally, write the full, coherent story based on your outline.

# Your output should be of the following format:

# Answers:
# Your answers here.

# Outline:
# Your 3 point outline here.

# Story:
# Your story here.
# '''

# double_check_questions_prompt = '''Think carefully about the answers to the following {n} questions: {questions}.

# Now go over the answers and double-check their factual correctness.

# Once you are confident in the answers, write a short and coherent story about {topic} that seamlessly incorporates these answers.

# First Answers:
# Initial answers to the questions here.

# Second Answers:
# Second try, double check the answers here.

# Story:
# Story here.
# '''



one_at_a_time_focus_prompt = '''Firstly, think carefully about each of the following {n} questions, focusing on one at a time: {questions}.

After you have thought through each question, write a short and coherent story about {topic} that incorporates the answers to all the questions. Make sure the story is engaging and that all trivia answers are factually correct.
'''

one_at_a_time_plan_prompt = '''Follow these steps precisely:

Step 1: Answer the following {n} questions one at a time: {questions}.

Step 2: Now double check each answer for factual correctness, one at a time.

Step 3: Once you are confident in the answers, create a brief, plan for a story about {topic} that logically incorporates all the answers.

Step 4: Finally, write the full, coherent story based on your outline.

Your output should be of the following format:

Answers:
Your answers here.

Double check answers:
Double check the answers here.

Plan:
Your plan here.

Story:
Your story here.
'''



conflict_plan_prompt = '''
Given a Creative Writing Task, describe the central conflict in detail (more than 5 sentences). The description should answer the following questions:
⋆ What’s the protagonist’s main goal in this story?
⋆ Why do they want it?
⋆ What’s stopping them from achieving it?
{scratchpad}

Central conflict:
'''

character_plan_prompt = '''
Given a Creative Writing Task and the Central Conflict, describe the characters in detailed bullet points (more than 5 sentences for each character). The description should answer the following questions:
⋆ What do the characters sound like? Are they talkative or quiet? What kind of slang do they use? What is their sense of humor like?
⋆ What do they look like? Do they have any defining gestures? What’s the first thing people notice about them?
⋆ What are their motivations and internal characteristics? What are their flaws? What are their values? What are they afraid of? How will they change and grow over the course of this story?
{scratchpad}

Character descriptions:
'''

setting_plan_prompt = '''
Given a Creative Writing Task, the Central Conflict and the Character Descriptions describe the setting in detail (more than 5 sentences). The description should answer the following questions:
⋆ Where does the story take place? Is it set in a fictional world, or is it simply set in someone’s backyard?
⋆ When does the story take place? What decade is it set in? How much time elapses over the course of the story?
{scratchpad}

Setting:
'''

plot_plan_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions and the Setting, describe the key plot points in detailed bullet points.
{scratchpad}

Key plot points:
'''

finalizer_writing_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions, the Setting and the Key Plot Points, write a story using the information below.
{scratchpad}
'''

exposition_writing_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions, the Setting and the Key Plot Points, continue the story by writing the {section} part.
Focus only on the {section} part of the story. Do not write about the following parts of the story. Do not end the story.
{scratchpad}

Exposition:
'''

rising_action_writing_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions, the Setting, the Key Plot Points and the Exposition, continue the story by writing the {section} part.
Begin your portion of the story in a way that naturally flows from the previous ending.
Match the writing style, vocabulary, and overall mood of the existing text. Do not re-explain details or events that have already been described.
Focus only on the {section} part of the story. Do not write about the following parts of the story. Do not end the story.
{scratchpad}

Rising action:
'''

climax_writing_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions, the Setting, the Key Plot Points, the Exposition and the Rising Action, continue the story by writing the {section} part.
Begin your portion of the story in a way that naturally flows from the previous ending.
Match the writing style, vocabulary, and overall mood of the existing text. Do not re-explain details or events that have already been described.
Focus only on the {section} part of the story. Do not write about the following parts of the story. Do not end the story.
{scratchpad}

Climax:
'''

falling_action_writing_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions, the Setting, the Key Plot Points, the Exposition, the Rising Action and the Climax, continue the story by writing the {section} part.
Begin your portion of the story in a way that naturally flows from the previous ending.
Match the writing style, vocabulary, and overall mood of the existing text. Do not re-explain details or events that have already been described.
Focus only on the {section} part of the story. Do not write about the following parts of the story. Do not end the story.
{scratchpad}

Falling action:
'''

resolution_writing_prompt = '''
Given a Creative Writing Task, the Central Conflict, the Character Descriptions, the Setting, the Key Plot Points, the Exposition, the Rising Action, the Climax and the Falling Action, continue the story by writing the {section} part.
Begin your portion of the story in a way that naturally flows from the previous ending.
Match the writing style, vocabulary, and overall mood of the existing text. Do not re-explain details or events that have already been described.
{scratchpad}

Resolution:
'''

write_standard = '''
Write a short and coherent story about {topic} that incorporates the following following words: {scratchpad}
'''







