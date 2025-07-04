import json
import time
import argparse

# Import the specific components from your other files
from tasks import trivia_creative_writing
from models import load_gemma_model, generate_text_with_gemma, get_probability_of_true
from base_agent import BaseAgent

class PlanningAgent(BaseAgent):
    def __init__(self, model, processor, device):
        super().__init__(model, processor, device)
        
    
        
    