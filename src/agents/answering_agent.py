import random

from .base_agent import BaseAgent

class AnsweringAgent(BaseAgent):
    def __init__(self, model, processor, task, device, scratchpad):
        super().__init__(model, processor, task, device, scratchpad)
        
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
        output = {}
        self.scratchpad = scratchpad
        qa_prompt = self.task.get_input_prompt(i, method, **kwargs) 
        question_answer_output = self.process_single_instance(model, processor, i, method, prompt=qa_prompt, test_output=True, **kwargs)
        # answers = question_answer_output['unwrapped_text']
        output["output"] = {
            "prompt": qa_prompt,
            "answers": question_answer_output["unwrapped_text"],
            "ground_truth": question_answer_output['evaluation']['ground_truth']
        }
        raw_generated_text = question_answer_output['raw_generated_text']
        self.scratchpad += f"[Words To Include] {raw_generated_text}"
        return output, question_answer_output['evaluation']
       
    def one_at_a_time_answer(self, model, processor, i, method, scratchpad, **kwargs):
        """
        Answer questions one at a time for the instance at index i.
        """
        print("\tidx:", i, "answering one at a time...")
        self.scratchpad = scratchpad
        output = {}
        answers = []
        question_prompts, questions = self.task.get_input_prompt(i, method, phase="question", **kwargs)
        # for question_prompt, question in zip(question_prompts, questions):
        j = 0
        for prompt in question_prompts:
            # question_answer_output = self.process_single_instance(model, processor, i, method, prompt=question_prompt, test_output=True, phase="question", **kwargs)
            question_answer_output = self.process_single_instance(model, processor, i, method, prompt=prompt, test_output=True, phase="question", **kwargs)
            # answers.append(question_answer_output["unwrapped_text"])
            output[f"{j}"] = {
                "prompt": prompt,
                "answers": question_answer_output["unwrapped_text"],
                "ground_truth": question_answer_output['evaluation']['ground_truth']
            }
            j += 1
            answers.append(question_answer_output["unwrapped_text"])
        words_to_include = ", ".join(output)
        self.scratchpad += f"[Words To Include] {words_to_include}"
        return output, answers, question_answer_output['evaluation']

            
        

        
    
    