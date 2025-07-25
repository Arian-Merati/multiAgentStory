import os
import re
from tasks.base import Task, DATA_PATH
from prompts import *
import json
# from models import gpt

class TriviaCreativeWritingTask(Task):
    def __init__(self, file='trivia_creative_writing_100_n_5.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'trivia_creative_writing', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx, method, **kwargs):
        datapoint = self.data[idx]
        questions = datapoint["questions"]
        topic = datapoint["topic"]
        n = len(questions)
        questions_str = " ".join(questions)
        
        if method == "standard":
            input_prompt = standard_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "cot":
            input_prompt = cot_prompt.format(n=n, questions=questions_str, topic=topic)
        # elif method == "spp":
        #     input_prompt = spp_prompt.format(n=n, questions=questions_str, topic=topic)
        # elif method == "spp_less_demo":
        #     input_prompt = spp_prompt_less_demo.format(n=n, questions=questions_str, topic=topic)
        # elif method == "spp_fixed_persona":
        #     input_prompt = spp_prompt_fixed_persona.format(n=n, questions=questions_str, topic=topic)
        # elif method == "spp_profile":
        #     input_prompt = spp_prompt_profile.format(n=n, questions=questions_str, topic=topic)
        # elif method == "questions_first":
        #     input_prompt = questions_first_prompt.format(n=n, questions=questions_str, topic=topic)
        # elif method == "persona":
        #     input_prompt = persona_prompt.format(n=n, questions=questions_str, topic=topic)
        # elif method == "structured_decomposition":
        #     input_prompt = structured_decomposition_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "double_check_questions":
            input_prompt = double_check_questions_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "double_check_one_at_a_time":
            input_prompt = double_check_one_at_a_time_prompt.format(question=kwargs['question'], proposed_answer=kwargs['proposed_answer'])
        # elif method == "one_at_a_time_focus":
        #     input_prompt = one_at_a_time_focus_prompt.format(n=n, questions=questions_str, topic=topic)
        # elif method == "one_at_a_time_plan":
        #     input_prompt = one_at_a_time_plan_prompt.format(n=n, questions=questions_str, topic=topic)
        # elif method == "self_refine":
        #     phase = kwargs["phase"]
        #     if phase == "init":
        #         input_prompt = standard_prompt.format(n=n, questions=questions_str, topic=topic)
        #     elif phase == "feedback":
        #         input_prompt = self_refine_feedback_prompt.format(question_answer=kwargs["question_answer"])
        #     elif phase == "refine":
        #         input_prompt = self_refine_refinement_prompt.format(question_answer=kwargs["question_answer"], feedback=kwargs["feedback"])
        elif method == "one_at_a_time_answer":
            input_prompt = confidence_assessment_question_prompt.format(question=kwargs['question'])
        elif method == "confidence_assessment":
            phase = kwargs["phase"]
            if phase == "question":
                input_prompts = []
                for question in questions:
                    input_prompts.append(confidence_assessment_question_prompt.format(question=question))
                return input_prompts, questions
            elif phase == "assess":
                input_prompt = confidence_assessment_assess_prompt.format(question=kwargs['question'], proposed_answer=kwargs['proposed_answer'])
            elif phase == 'write':
                input_prompt = confidence_assessment_write_prompt.format(topic=topic, n=n, answers=kwargs['answers'])
        else:
            raise NotImplementedError(f"method {method} not implemented")
        
        return input_prompt

    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia questions
        instance = self.data[idx]
        correct_count = 0
        question_count = len(instance["answers"])
        for ans_to_question in instance["answers"]:
            for ans in ans_to_question:
                # compare all to lower
                if ans.lower() in output.lower():
                    correct_count += 1
                    break
        info = {'correct_count': correct_count, 'question_count': question_count}
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        if method in ["standard", "self_refine", "persona", "one_at_a_time_focus"]:
            return response, True
        
        elif method in ["cot", "one_at_a_time_plan", "double_check_questions", "structured_decomposition", "questions_first"]:
            if "Story:" in response:
                return response.split("Story:")[1].strip(), True
            elif "story:" in response:
                return response.split("story:")[1].strip(), True
            else:
                return response, False
        
        elif method in ["spp","spp_profile","spp_fixed_persona", "spp_less_demo"]:
            if "Final answer:" in response:
                return response.split("Final answer:")[1].strip(), True
            elif "final answer:" in response:
                return response.split("final answer:")[1].strip(), True
            else:
                return response, False 
            
        elif method == "double_check_one_at_a_time":
            if "Revised answer:" in response:
                return response.split("Revised answer:")[1].strip(), True
            elif "revised answer:" in response:
                return response.split("revised answer:")[1].strip(), True
            else:
                return response, False 
            
        elif method == "confidence_assessment":
            phase = kwargs["phase"]
            if phase == "question":
                if "Answer:" in response:
                    return response.split("Answer:")[1].strip(), True
                elif "answer:" in response:
                    return response.split("answer:")[1].strip(), True
                else:
                    return response, False
            elif phase == "assess":
                if "The proposed answer is:" in response:
                    return response.split("The proposed answer is: (")[1].strip(), True
                elif "the proposed answer is:" in response:
                    return response.split("the proposed answer is: (")[1].strip(), True
                else:
                    return response, False
            elif phase == 'write':
                return response, True
            
        elif method == "plan_ar":
            phase = kwargs["phase"]
            if phase == "central_conflict":
                if "Central conflict:" in response:
                    return response.split("Central conflict:")[1].strip(), True
                elif "central conflict" in response:
                    return response.split("central conflict:")[1].strip(), True
                else:
                    return response, False
            elif phase == "character":
                if "Character descriptions:" in response:
                    return response.split("Character descriptions:")[1].strip(), True
                elif "character descriptions" in response:
                    return response.split("character descriptions:")[1].strip(), True
                else:
                    return response, False
            elif phase == "setting":
                if "Setting:" in response:
                    return response.split("Setting:")[1].strip(), True
                elif "setting:" in response:
                    return response.split("setting:")[1].strip(), True
                else:
                    return response, False
            elif phase == "plot":
                if "Key plot points:" in response:
                    return response.split("Key plot points:")[1].strip(), True
                elif "key plot points:" in response:
                    return response.split("key plot points:")[1].strip(), True
                else:
                    return response, False
                
        elif method == "write_ar":
            phase = kwargs["phase"]
            if phase == "exposition":
                if "Exposition:" in response:
                    return response.split("Exposition:")[1].strip(), True
                elif "the proposed answer is:" in response:
                    return response.split("exposition:")[1].strip(), True
                else:
                    return response, False
            elif phase == "rising_action":
                if "Rising action:" in response:
                    return response.split("Rising action:")[1].strip(), True
                elif "rising action:" in response:
                    return response.split("rising action:")[1].strip(), True
                else:
                    return response, False
            elif phase == "climax":
                if "Climax:" in response:
                    return response.split("Climax:")[1].strip(), True
                elif "climax:" in response:
                    return response.split("climax:")[1].strip(), True
                else:
                    return response, False
            elif phase == "falling_action":
                if "Falling action:" in response:
                    return response.split("Falling action:")[1].strip(), True
                elif "falling action:" in response:
                    return response.split("falling actions:")[1].strip(), True
                else:
                    return response, False
            elif phase == "resolution":
                if "Resolution:" in response:
                    return response.split("Resolution:")[1].strip(), True
                elif "resolution:" in response:
                    return response.split("resolution:")[1].strip(), True
                else:
                    return response, False
            
        else:
            raise NotImplementedError(f"method {method} not implemented")