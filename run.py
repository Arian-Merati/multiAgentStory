import json
import yaml
import re
from src.agents import *
import argparse

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true

# MODEL_CONFIG = {
#     "model_id": "google/gemma-3-4b-it",
#     "device": "mps",
# }

AGENT_MAPPING = {
    "answer_all": AnsweringAgent,
    "answer_one_at_a_time": AnsweringAgent,
    # "gold_label": AnsweringAgent,
    # "confidence_assessment": CheckingAgent,
    # "double_check_all": CheckingAgent,
    # "double_check_oaat": CheckingAgent,
    # "plan_AR": PlanningAgent,
    # "plan_cot": PlanningAgent,
    # "write_AR": WritingAgent,
    # "write_standard": WritingAgent,
}

TASK_FILE = "trivia_creative_writing_100_n_5.jsonl"
# SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

def save_progress(logs, output_file):
    with open(output_file, "w") as f:
        json.dump(logs, f, indent=4)
        
def get_identifiers(scratchpad):
    """
    Extract identifiers from the scratchpad.
    """
    identifiers = re.findall(r'\[(.*?)\]', scratchpad)
    if not identifiers:
        return None
    last_identifier = identifiers.pop()
    identifiers_str = ", the ".join(identifiers)
    identifiers_str += " and the " + last_identifier 
    
    return identifiers_str
    

def main():
    
    parser = argparse.ArgumentParser(description="Run multi-agent experiments with Gemma.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the output files.")
    args = parser.parse_args()

    model, processor = load_gemma_model(
        model_id=args.model_path,
        device=args.device,
    )

    task = trivia_creative_writing.TriviaCreativeWritingTask(file=TASK_FILE)
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    output = {}
    experiments = config['experiments']
    for experiment_name, agent_list in experiments.items():
        print(f"Running experiment: {experiment_name}")
        results = {}
        for i in range (len(task)):
            if i != 55:
                continue
            scratchpad = ""
            answers = None
            evaluation = None
            
            for agent_name in agent_list:
                agent_class = AGENT_MAPPING[agent_name]
                agent = agent_class(model, processor, args.device, scratchpad=scratchpad)
                
                # identifiers = get_identifiers(scratchpad)
               
                # if agent_name == 'gold_label':
                #     answers = agent.gold_label(i, scratchpad)
                if agent_name == "answer_all":
                    answers, evaluation = agent.answer_all(model, processor, i, method="answer_all", scratchpad=scratchpad)
                elif agent_name == 'answer_one_at_a_time':
                    answers, evaluation = agent.one_at_a_time_answer(model, processor, i, method="confidence_assessment", scratchpad=scratchpad)
                # elif agent_name == "double_check_one_at_a_time":
                #     revised_answers, answers_str = agent.double_check_one_at_a_time(model, processor, task, i, method="confidence_assessment", proposed_answers_list=answers, scratchpad=scratchpad)
                # elif agent_name == "double_check_all":
                #     revised_answers = agent.double_check(model, processor, task, i, method="double_check_all", scratchpad=scratchpad)
                # elif agent_name == "confidence_assessment":
                #     log_outputs = agent.confidence_assessment(model, processor, task, i, method="confidence_assessment", scratchpad=scratchpad)
                #     answers = log_outputs["answers"]
                # elif agent_name == "plan_ar":
                #     agent.plan_ar(i, scratchpad=scratchpad, identifiers=identifiers)
                # elif agent_name == "plan_standard":
                #     agent
                
            results[i] = {
                "evaluation": evaluation,
                "outputs": answers
            }
   
        f1_sum = 0
        correct_count_sum = 0
        question_count_sum = 0
        
        f1_full_list = [f1_score for result in results.values() for f1_score in result['evaluation']['f1']]
        f1_sum = sum(f1_full_list)
        avg_f1 = f1_sum / len(f1_full_list)
        
        for i, result in results.items():
            correct_count_sum += result['evaluation']['correct_count']
            question_count_sum += result['evaluation']['question_count']

        avg_accuracy = correct_count_sum / question_count_sum
        
        print(f"Average F1 score for {experiment_name}: {avg_f1:.2f}")
        print(f"Average accuracy for {experiment_name}: {avg_accuracy:.2f}")
        
        output[experiment_name] = {
            "f1": avg_f1,
            "accuracy": avg_accuracy,
            "results": results
        }
        
        output_file = f"{args.output_dir}/TEST_{experiment_name}.jsonl"
        save_progress(output, output_file)
        
if __name__ == "__main__":
    main()