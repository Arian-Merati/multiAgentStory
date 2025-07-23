import json
import time
import argparse
import random

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true
from base_agent import BaseAgent

class AnsweringAgent(BaseAgent):
    def __init__(self, model, processor, device, scratchpad=""):
        super().__init__(model, processor, device, scratchpad="")
        
    def gold_label(self, i, scratchpad):
        """
        Get the gold label for the instance at index i.
        """
        self.scratchpad = scratchpad
        random_answers = []
        data = self.task.get_input(i)
        answers = data['answers']
        for answer_list in answers:
            random_choice = random.choice(answer_list)
            # Add the chosen answer to our new list
            random_answers.append(random_choice)
        words_to_include = random_answers.join(", ")
        scratchpad += f"[Words To Include] {words_to_include}"
        return random_answers
       
    
    def one_at_a_time_answer(self, model, processor, task, i, method, scratchpad, prompt=None, test_output=False, **kwargs):
        """
        Answer questions one at a time for the instance at index i.
        """
        print("\tidx:", i, "answering one at a time...")
        self.scratchpad = scratchpad
        answers = []
        question_prompts, questions = self.task.get_input_prompt(i, method=method, phase="question", **kwargs)
        for question_prompt, question in zip(question_prompts, questions):
            question_answer_output = self.process_single_instance(model, processor, task, i, method=method, prompt=question_prompt, test_output=False)
            answers.append(question_answer_output["unwrapped_text"])
        words_to_include = answers.join(", ")
        scratchpad += f"[Words To Include] {words_to_include}"
        return answers

            
        

        
    
    