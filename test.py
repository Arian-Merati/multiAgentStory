import json
import time
import argparse

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma

MODEL_CONFIG = {
    "model_id": "google/gemma-3-4b-it",
    "device": "mps",
}
TASK_FILE = "trivia_creative_writing_100_n_5.jsonl"
METHOD = "standard"
OUTPUT_LOG_FILE = f"gemma_trivia_results__{METHOD}.jsonl"


def save_progress(logs, output_file):
    with open(output_file, "w") as f:
        json.dump(logs, f, indent=4)

def process_single_instance(model, processor, task, i, method, prompt=None, test_output=True):
    if prompt is None:
        prompt = task.get_input_prompt(i, method=method)
    raw_generated_text = generate_text_with_gemma(model, processor, prompt, MODEL_CONFIG["device"])
    unwrapped_text, _ = task.prompt_unwrap(raw_generated_text, method=method)
    if test_output:
        eval_info = task.test_output(i, unwrapped_text)
    else:
        eval_info = []
    
    return {
        "idx": i,
        "method": method,
        "evaluation": eval_info,
        "unwrapped_text": unwrapped_text,
        "raw_generated_text": raw_generated_text,
        "prompt": prompt,
    }

def run_default(model, processor, task, method, output_file):
    print(f"\n--- Starting Standard Evaluation on {len(task)} Instances ---")
    all_logs = []
    start_time = time.time()

    for i in range(len(task)):
        print(f"Processing instance {i + 1}/{len(task)}...")
        log_entry = process_single_instance(model, processor, task, i, method)
        print(f"  Result: {log_entry['evaluation']['correct_count']} / {log_entry['evaluation']['question_count']} correct")
        all_logs.append(log_entry)
        save_progress(all_logs, output_file)

    return all_logs

# def run_self_refine(model, processor, task, method, output_file, system_prompt, num_refine=1):
#     print(f"\n--- Starting Self-Refine Evaluation on {len(task)} Instances ---")
#     all_logs = []
#     start_time = time.time()
#     init_method = 'standard'

#     for i in range(len(task)):
#         print(f"Processing instance {i + 1}/{len(task)}...")
#         log_outputs = {}
        
#         init_prompt = task.get_input_prompt(i, method=method, phase="init")
#         init_log = process_single_instance(model, processor, task, i, init_method, system_prompt)
#         if not init_log:
#             continue
#         log_outputs["answer_0"] = init_log
        
#         # Build context from initial response
#         context_prompt = init_log['prompt'] + "\n" + init_log['raw_generated_text']
        
#         # Refinement iterations
#         for j in range(num_refine):
#             print(f"  Refinement step {j + 1}/{num_refine}")
            
#             # Get feedback
#             feedback_prompt = task.get_input_prompt(i, method=method, phase="feedback", question_answer=context_prompt)
#             feedback_log = process_single_instance(model, processor, task, i, method, system_prompt)
#             if not feedback_log:
#                 break
#             log_outputs[f"feedback_{j}"] = feedback_log
            
#             # Get refined response
#             refine_prompt = task.get_input_prompt(i, method=method, phase="refine", 
#                                                 question_answer=context_prompt, 
#                                                 feedback=feedback_log['unwrapped_text'])
#             refine_log = process_single_instance(model, processor, task, i, method, system_prompt, prompt=refine_prompt)
#             if not refine_log:
#                 break
#             log_outputs[f"answer_{j+1}"] = refine_log
            
#             # Update context for next iteration
#             context_prompt = refine_prompt + "\n" + refine_log['raw_generated_text']
        
#         all_logs.append({
#             "idx": i,
#             "method": method,
#             "refinement_logs": log_outputs
#         })
#         save_progress(all_logs, output_file)

#     return all_logs


def run_self_refine(model, processor,task, i, method, num_refine=1, **kwargs):
    print("\tidx:", i, "start self refine...")
    log_outputs = {}
    ## get initial response
    init_prompt = task.get_input_prompt(i, method=method, phase="init", **kwargs)
    init_output = process_single_instance(model, processor, task, i, method, prompt=init_prompt, test_output=True)
    if init_output == {}:
        return {}
    log_outputs["answer_0"] = init_output

    context_prompt = init_prompt + "\n" + init_output["raw_generated_text"] # Q + A0
    for j in range(num_refine):
        print("\t\tstep:", j)
        # get feedback
        feedback_prompt = task.get_input_prompt(i, method=method, phase="feedback", question_answer=context_prompt, **kwargs)
        feedback_output = process_single_instance(model, processor, task, i, method, prompt=feedback_prompt, test_output=False)
        if feedback_output == {}:
            return log_outputs
        log_outputs[f"feedback_{j}"] = feedback_output


        # get refined response
        refine_prompt = task.get_input_prompt(i, method=method, phase="refine", question_answer=context_prompt, feedback=feedback_output["unwrapped_text"], **kwargs) # Q + A0 + F
        refine_output = process_single_instance(model, processor, task, i, method, prompt=refine_prompt, test_output=True)
        if refine_output == {}:
            return log_outputs
        log_outputs[f"answer_{j+1}"] = refine_output

        # update context
        context_prompt = refine_prompt + refine_output["raw_generated_text"] # Q + A0 + F + A1

    return log_outputs

def run_confidence_assessment(model, processor,task, i, method, **kwargs):
    print("\tidx:", i, "start self refine...")
    log_outputs = {}
    question_prompt = task.get_input_prompt(i, method=method, phase="question", **kwargs)
    question_output = process_single_instance(model, processor, task, i, method, prompt=question_prompt, test_output=False)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='standard')
    parser.add_argument('--num_refine', type=int, default=1)
    parser.add_argument('--output_file', type=str, default=OUTPUT_LOG_FILE)
    args = parser.parse_args()

    model, processor = load_gemma_model(
        model_id=MODEL_CONFIG["model_id"],
        device=MODEL_CONFIG["device"],
    )

    task = trivia_creative_writing.TriviaCreativeWritingTask(file=TASK_FILE)

    # Run appropriate evaluation
    if args.method == 'self_refine':
        run_self_refine(model, processor, task, args.method, args.output_file, args.num_refine)
    else:
        run_default(model, processor, task, args.method, args.output_file)

if __name__ == "__main__":
    main()