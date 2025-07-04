import json
import time
import argparse
import random

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true
from base_agent import BaseAgent

class AnsweringAgent(BaseAgent):
    def __init__(self, model, processor, device):
        super().__init__(model, processor, device)
        
    def gold_label(self, i, method, **kwargs):
        """
        Get the gold label for the instance at index i.
        """
        random_answers = []
        data = task.get_input(i, method=method, **kwargs)
        answers = data['answers']
        for answer_list in answers:
            random_choice = random.choice(answer_list)
            # Add the chosen answer to our new list
            random_answers.append(random_choice)
        
        return random_answers.join(", ")
    
    
    def one_at_a_time_answer(self, model, processor, task, i, method, prompt=None, test_output=False, **kwargs):
        """
        Double-check the output of a single instance, one question at a time.
        """
        print("\tidx:", i, "answering one at a time...")
        answers = []
        question_prompts, questions = task.get_input_prompt(i, method=method, phase="question", **kwargs)
        for question_prompt, question in zip(question_prompts, questions):
            question_answer_output = self.process_single_instance(model, processor, task, i, method, prompt=checking_prompt, test_output=False, phase="assess")
            answers.append(question_answer_output["unwrapped_text"])
        answers_str = " ".join(answers)
           
        return answers, answers_str
            
        

        
    
    