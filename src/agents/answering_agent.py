import random

from .base_agent import BaseAgent

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
        words_to_include = ", ".join(random_answers)
        self.scratchpad += f"[Words To Include] {words_to_include}"
        return random_answers
    
    def answer_all(self, model, processor, i, method, scratchpad, **kwargs):
        """
        Answer all questions at once
        """
        self.scratchpad = scratchpad
        qa_prompt = self.task.get_input_prompt(i, method, **kwargs) 
        question_answer_output = self.process_single_instance(model, processor, i, method, prompt=qa_prompt, test_output=True, **kwargs)
        answers = question_answer_output['unwrapped_text']
        raw_generated_text = question_answer_output['raw_generated_text']
        self.scratchpad += f"[Words To Include] {raw_generated_text}"
        return answers, question_answer_output['evaluation']
       
    def one_at_a_time_answer(self, model, processor, i, method, scratchpad, **kwargs):
        """
        Answer questions one at a time for the instance at index i.
        """
        print("\tidx:", i, "answering one at a time...")
        self.scratchpad = scratchpad
        answers = []
        question_prompts, questions = self.task.get_input_prompt(i, method, phase="question", **kwargs)
        # for question_prompt, question in zip(question_prompts, questions):
        for prompt in question_prompts:
            # question_answer_output = self.process_single_instance(model, processor, i, method, prompt=question_prompt, test_output=True, phase="question", **kwargs)
            question_answer_output = self.process_single_instance(model, processor, i, method, prompt=prompt, test_output=True, phase="question", **kwargs)
            answers.append(question_answer_output["unwrapped_text"])
        words_to_include = ", ".join(answers)
        self.scratchpad += f"[Words To Include] {words_to_include}"
        return answers, question_answer_output['evaluation']

            
        

        
    
    