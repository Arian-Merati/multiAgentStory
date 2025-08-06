import json
import time
import argparse

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true
from .base_agent import BaseAgent

class CheckingAgent(BaseAgent):
    def __init__(self, model, processor, task, device, scratchpad):
        super().__init__(model, processor, task, device, scratchpad)
        
    # def double_check(self, model, processor, i, method, scratchpad, proposed_answers, **kwargs):
    #     """
    #     Double-check the output of a single instance.
    #     """
    #     answers = {}
    #     double_check_prompt = self.task.get_input_prompt(i, method="double_check", **kwargs) 
    #     checked_output = self.process_single_instance(model, processor, i, method, prompt=double_check_prompt, test_output=True, **kwargs)
    #     answers["output"] = {
    #         "prompt": double_check_prompt,
    #         "answer": checked_output["unwrapped_text"],
    #         "raw_generated_text": checked_output['raw_generated_text'],
    #         "ground_truth": checked_output['evaluation']['ground_truth']
    #     }
    #     # verified_answer = checked_output['unwrapped_text']
    #     # raw_generated_text = checked_output['raw_generated_text']
    #     self.scratchpad += f"[Words To Include] {checked_output['raw_generated_text']}"
    #     return answers, checked_output['evaluation']
    
    def confidence_assessment(self, model, processor, i, method, scratchpad, **kwargs):
        """
        Double-check the output of a single instance, one question at a time.
        """
        print("\tidx:", i, "start confidence assessment...")
        self.scratchpad = scratchpad
        log_outputs = {}
        answers = []
        j = 0
        question_prompts, questions = self.task.get_input_prompt(i, method=method, phase="question", **kwargs)
        for question_prompt, question in zip(question_prompts, questions):
            confidence = 0
            assessment_prompts = []
            while confidence <= 0.9:
                question_output = self.process_single_instance(model, processor, i, method, prompt=question_prompt, test_output=False, phase="question")
                assessment_prompt = self.task.get_input_prompt(i, method=method, phase="assess", question=question, proposed_answer=question_output["unwrapped_text"])
                assessment_prompts.append(assessment_prompt)
                confidence = get_probability_of_true(model, processor, assessment_prompt, self.device)
                print(f"Confidence for question '{question}': {confidence:.2f}")
            j += 1
            log_outputs[f"assessment_prompt{j}"] = assessment_prompts
            answers.append(question_output["unwrapped_text"])
        answers_str = " ".join(answers)
        write_prompt = self.task.get_input_prompt(i, method=method, phase='write', answers=answers_str)
        write_output = self.process_single_instance(model, processor, i, method, prompt=write_prompt, test_output=True, phase="write")
        log_outputs["answers"] = answers
        log_outputs["question_prompts"] = question_prompts
        log_outputs["write_prompt"] = write_prompt
        log_outputs["write_output"] = write_output
        
        answers_str = " ".join(answers)
        self.scratchpad = f"[Words To Include] {answers_str}"
            
        return log_outputs, write_output["evaluation"]
    
    
    def double_check(self, model, processor, i, method, scratchpad, proposed_answers_list, **kwargs):
        """
        Double-check the output of a single instance, one question at a time.
        """
        print("\tidx:", i, "double check one at a time...")
        self.scratchpad = scratchpad
        question_prompts, questions_list = self.task.get_input_prompt(i, method="confidence_assessment", phase="question", **kwargs)
        revised_answers = []
        output = {}
        for j, (question, proposed_answer) in enumerate(zip(questions_list, proposed_answers_list)):
            print(f"proposed_answer: {proposed_answer}")
            print(f"quesiton: {question}")
            checking_prompt = self.task.get_input_prompt(i, method=method, question=question, proposed_answer=proposed_answer)
            print(f"checking prompt: {checking_prompt}")
            double_check_output = self.process_single_instance(model, processor, i, method, prompt=checking_prompt, **kwargs)
            output[f"{j}"] = {
                "prompt": checking_prompt,
                "answer": double_check_output["unwrapped_text"],
                "raw_generated_text": double_check_output['raw_generated_text'],
                "ground_truth": double_check_output['ground_truth'],
                "evaluation": double_check_output['evaluation']
            }
            revised_answers.append(double_check_output["unwrapped_text"])
        answers_str = " ".join(revised_answers)
        self.scratchpad = f"[Words To Include] {answers_str}"
           
        return output
    
    
    

        
    
    
    

        