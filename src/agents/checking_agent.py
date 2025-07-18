import json
import time
import argparse

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true
from base_agent import BaseAgent

class CheckingAgent(BaseAgent):
    def __init__(self, model, processor, device, scratchpad=""):
        super().__init__(model, processor, device, scratchpad="")
        
    def double_check(self, model, processor, task, i, method, scratchpad, prompt=None, test_output=False, **kwargs):
        """
        Double-check the output of a single instance.
        """
        double_check_prompt = task.get_input_prompt(i, method="double_check", **kwargs) 
        checked_output = self.process_single_instance(model, processor, task, i, method, prompt=double_check_prompt, test_output=False, **kwargs)
        verified_answer = checked_output['unwrapped_text']
        raw_generated_text = checked_output['raw_generated_text']
        self.scratchpad += f"\n\n[Checking Answers To Questions] {raw_generated_text}"
        return verified_answer
    
    def confidence_assessment(self, model, processor, task, i, method, scratchpad, prompt=None, test_output=False, **kwargs):
        """
        Double-check the output of a single instance, one question at a time.
        """
        print("\tidx:", i, "start confidence assessment...")
        log_outputs = {}
        answers = []
        j = 0
        question_prompts, questions = task.get_input_prompt(i, method=method, phase="question", **kwargs)
        for question_prompt, question in zip(question_prompts, questions):
            confidence = 0
            assessment_prompts = []
            while confidence <= 0.5:
                question_output = self.process_single_instance(model, processor, task, i, method, prompt=question_prompt, test_output=False, phase="question")
                assessment_prompt = task.get_input_prompt(i, method=method, phase="assess", question=question, proposed_answer=question_output["unwrapped_text"])
                assessment_prompts.append(assessment_prompt)
                confidence = get_probability_of_true(model, processor, assessment_prompt, MODEL_CONFIG["device"])
                print(f"Confidence for question '{question}': {confidence:.2f}")
            j += 1
            log_outputs[f"assessment_prompt{j}"] = assessment_prompts
            answers.append(question_output["unwrapped_text"])
        answers_str = " ".join(answers)
        write_prompt = task.get_input_prompt(i, method=method, phase='write', answers=answers_str)
        write_output = self.process_single_instance(model, processor, task, i, method, prompt=write_prompt, test_output=True, phase="write")
        log_outputs["answers"] = answers
        log_outputs["question_prompts"] = question_prompts
        log_outputs["write_prompt"] = write_prompt
        log_outputs["write_output"] = write_output
            
        return log_outputs
    
    
    def double_check_one_at_a_time(self, model, processor, task, i, method, scratchpad, proposed_answers_list, prompt=None, test_output=False, **kwargs):
        """
        Double-check the output of a single instance, one question at a time.
        """
        print("\tidx:", i, "double check one at a time...")
        question_prompts, questions_list = task.get_input_prompt(i, method="confidence_assessment", phase="question", **kwargs)
        revised_answers = []
        for question, proposed_answer in zip(questions_list, proposed_answers_list):
            checking_prompt = task.get_input_prompt(i, method=method, question=question, proposed_answer=proposed_answer)
            double_check_output = self.process_single_instance(model, processor, task, i, method, prompt=checking_prompt, test_output=False, **kwargs)
            revised_answers.append(double_check_output["unwrapped_text"])
        answers_str = " ".join(revised_answers)
        self.scratchpad = f"\n\n[Words To Include] {answers_str}"
           
        return revised_answers, answers_str
    
    
    

        
    
    
    

        