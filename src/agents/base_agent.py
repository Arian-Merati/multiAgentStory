import json
import time
import argparse

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true


MODEL_CONFIG = {
    "model_id": "google/gemma-3-4b-it",
    "device": "mps",
}
TASK_FILE = "trivia_creative_writing_100_n_5.jsonl"
# SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

class BaseAgent:
    def __init__(self, model_id=MODEL_CONFIG["model_id"], device=MODEL_CONFIG["device"], scratchpad=""):
        self.model, self.processor = load_gemma_model(model_id, device)
        self.task = trivia_creative_writing.TriviaCreativeWritingTask(file=TASK_FILE)
        self.scratchpad = scratchpad
        
    def process_single_instance(model, processor, task, i, method, prompt=None, test_output=True, **kwargs):
        if prompt is None:
            prompt = task.get_input_prompt(i, method=method, **kwargs)
        raw_generated_text = generate_text_with_gemma(model, processor, prompt, MODEL_CONFIG["device"])
        unwrapped_text, _ = task.prompt_unwrap(raw_generated_text, method=method, **kwargs)
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