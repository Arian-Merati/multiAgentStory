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
    def __init__(self, model, processor, task, device, scratchpad=""):
        self.model = model
        self.processor = processor
        self.device = device
        self.task = task
        self.scratchpad = scratchpad
        
    def process_single_instance(self, model, processor, i, method, **kwargs):
        test_output = kwargs.get("test_output", True)
        prompt = kwargs.get("prompt", None)
        if prompt is None:
            prompt = self.task.get_input_prompt(i, method, **kwargs)
        raw_generated_text = generate_text_with_gemma(model, processor, prompt, MODEL_CONFIG["device"])
        unwrapped_text, _ = self.task.prompt_unwrap(raw_generated_text, method=method, **kwargs)
        if test_output:
            eval_info = self.task.test_output(i, unwrapped_text)
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