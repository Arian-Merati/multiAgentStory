import os
import re
from tasks.base import Task, DATA_PATH
from prompts import *
import json
import collections
# from models import gpt

class TriviaCreativeWritingTask(Task):
    def __init__(self, file='trivia_creative_writing_100_n_5.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'trivia_creative_writing', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]
            
    def __len__(self):
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx, scratchpad, method, **kwargs):
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
        elif method == "double_check_all":
            input_prompt = double_check_all_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "double_check":
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
        # elif method == "one_at_a_time_answer":
        #     input_prompt = confidence_assessment_question_prompt.format(question=kwargs['question'])
        elif method == "answer_all":
            input_prompt = answer_all_prompt.format(n=n, questions=questions_str)
        elif method == "write_standard":
            input_prompt = write_standard.format(topic=topic, scratchpad=scratchpad)
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
    
    @staticmethod
    def f1_score(proposed_answer, gold_answers):
        """
        Calculate F1 score between proposed answer and gold answers.
        """
        f1 = 0
        best_f1_precision = 0
        best_f1_recall = 0
        
        # print(gold_answers)
        # print(proposed_answer)
        
        for answer in gold_answers:
            answer_tokens = answer.split()
            proposed_tokens = proposed_answer.split()
            common = collections.Counter(answer_tokens) & collections.Counter(proposed_tokens)
            num_same = sum(common.values())
            #print(num_same)
            precision = 1.0 * num_same / len(proposed_tokens)
            recall = 1.0 * num_same / len(answer_tokens)
            if precision + recall == 0:
                f1_temp = 0
            else:
                f1_temp = (2 * precision * recall) / (precision + recall)
            
            if f1_temp > f1:
                f1 = f1_temp
                best_f1_precision = precision
                best_f1_recall = recall
            
        return best_f1_precision, best_f1_recall, f1
        

    def test_output(self, idx, output):
        # test whether the output includes all the answers of the trivia questions
        instance = self.data[idx]
        correct_count = 0
        question_count = len(instance["answers"])
        f1_list = []
        
        # instance["answers"] is a list of lists containing all the answers to the questions at index i
        for ans_to_question in instance["answers"]:
            precision, recall, f1 = self.f1_score(output, ans_to_question)
            f1_list.append(f1)
            for ans in ans_to_question:
                # compare all to lower
                # print(f"ground truth answer: {ans.lower()}")
                # print(f"output: {output.lower()}")
                pattern = r'\b' + re.escape(str(ans).lower()) + r'\b'
                
                if re.search(pattern, output.lower()):
                    print(f"MATCH: Found answer '{ans}' in output.")
                    correct_count += 1
                    break 
                
                # if ans.lower() in output.lower():
                #     print("MATCH")
                #     correct_count += 1
                #     break
        
        info = {'correct_count': correct_count, 'question_count': question_count, 
                'accuracy': correct_count / question_count, 'f1': f1_list}
        
        # print(info)
        
        return info, instance["answers"]

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        if method in ["standard", "self_refine", "persona", "one_at_a_time_focus", "write_standard"]:
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
            
        elif method == "double_check":
            if "Revised answer:" in response:
                return response.split("Revised answer:")[1].strip(), True
            elif "revised answer:" in response:
                return response.split("revised answer:")[1].strip(), True
            else:
                return response, False 
            
        elif method == "answer_all":
            if "Answers:" in response:
                return response.split("Answers:")[1].strip(), True
            elif "answers:" in response:
                return response.split("answers:")[1].strip(), True
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
            if phase == "conflict":
                if "Central conflict:" in response:
                    return response.split("Central conflict:")[1].strip(), True
                elif "central conflict" in response:
                    return response.split("central conflict:")[1].strip(), True
                else:
                    return response, False
            elif phase == "characters":
                print("response:", response)
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
                    return response.split("falling action:")[1].strip(), True
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