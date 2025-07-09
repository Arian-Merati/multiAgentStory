import json
import time
import argparse
import yaml

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true

MODEL_CONFIG = {
    "model_id": "google/gemma-3-4b-it",
    "device": "mps",
}
TASK_FILE = "trivia_creative_writing_100_n_5.jsonl"
# SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

def save_progress(logs, output_file):
    with open(output_file, "w") as f:
        json.dump(logs, f, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='standard')
    parser.add_argument('--num_refine', type=int, default=1)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    # Set output file based on method if not specified
    if args.output_file is None:
        args.output_file = f"gemma_trivia_results__{args.method}.jsonl"

    model, processor = load_gemma_model(
        model_id=MODEL_CONFIG["model_id"],
        device=MODEL_CONFIG["device"],
    )

    task = trivia_creative_writing.TriviaCreativeWritingTask(file=TASK_FILE)
    
    with open(config.yaml, 'r') as f:
        config = yaml.safe_load(f)
        
    experiments = config['experiments']
    for experiment_name, agent_list in experiments.items():
        print(f"Running experiment: {experiment_name}")
        scratchpad = ""
        for agent in agent_list:
            print(f"  Agent: {agent}")
            
        
        

    # Run appropriate evaluation
    if args.method == 'self_refine':
        run_self_refine(model, processor, task, args.method, args.output_file, args.num_refine)
    elif args.method == 'confidence_assessment':
        run_confidence_assessment(model, processor, task, args.method, args.output_file)
    else:
        run_default(model, processor, task, args.method, args.output_file)

if __name__ == "__main__":
    main()