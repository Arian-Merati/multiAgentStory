import json

prompts = ['standard', 'cot', 'self_refine', 'persona', 'structured_decomposition', 'double_check_questions', 'one_at_a_time_focus', 'one_at_a_time_plan', 'questions_first']
results = {}


def main():
    
    for prompt in prompts:
        print(f"Running evaluation for prompt: {prompt}")
        correct_count = 0
        question_count = 0
        data = []
        if prompt == "self_refine":
            with open("gemma_trivia_results__self_refine.jsonl", "r") as f:
                data_list = json.load(f)
                for entry_json in data_list:
                    # print(json.dumps(entry_json, indent=4))
                    question_count += entry_json['refinement_logs']['answer_1']['evaluation']['question_count']
                    correct_count += entry_json['refinement_logs']['answer_1']['evaluation']['correct_count']
                metric = correct_count / question_count 
                results['self_refine'] = metric
        else:
            with open(f"gemma_trivia_results__{prompt}.jsonl", "r") as f:
                data_list = json.load(f)
                for entry_json in data_list:
                    # print(json.dumps(entry_json, indent=4))
                    question_count += entry_json['evaluation']['question_count']
                    correct_count += entry_json['evaluation']['correct_count']
                metric = correct_count / question_count 
                results[prompt] = metric
            
    standard_score = results['standard']
    for key, value in results.items():
        print(f"{key}: {value:.4f}")  
        difference = value - standard_score
        delta = (difference / standard_score) * 100 
        print(f"{key} is {delta:.2f}% {'higher' if delta > 0 else 'lower'} than standard.")       
                
    print(f"Evaluation for {prompt} completed.\n")
        
        
if __name__ == "__main__":
    main()