from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch.nn.functional as F
from PIL import Image
import requests
import torch

SYSTEM_PROMPT = "You are an AI assistant that helps people find information."


def load_gemma_model(model_id, device):
    """
    Load the Gemma model with specified device and dtype.

    Args:
        model_id (str): The identifier for the Gemma model.
        device (str): The device to load the model onto (e.g., 'cpu', 'cuda', 'mps').

    Returns:
        model (Gemma3ForConditionalGeneration): The loaded Gemma model.
        processor (AutoProcessor): The processor for the model.
   
    """
    model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded successfully.")
    return model, processor


def generate_text_with_gemma(model, processor, prompt, device):
    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    },
    {
        "role": "user",
        "content": [
            #{"type": "text", "text": "Write a short and coherent story about Superman that incorporates the answers to the following 5 questions: What was Warren Beatty's first movie? Which future Hollywood star got her break as Wonder Girl, Wonder Woman's sister Drusilla? Mickey Braddock in the 50s series Circus Boy found fame with which surname in which pop band? Which war veteran was Director of News & Special Events for ABC before find fame as a TV cop? What US sitcom was the first exported to Britain?"},
            #{"type": "text", "text": "Answer the following five questions first: What was Warren Beatty's first movie? Which future Hollywood star got her break as Wonder Girl, Wonder Woman's sister Drusilla? Mickey Braddock in the 50s series Circus Boy found fame with which surname in which pop band? Which war veteran was Director of News & Special Events for ABC before find fame as a TV cop? What US sitcom was the first exported to Britain? Now incorporate them into a short and coherent story about Superman."},
            {"type": "text", "text": prompt},
        ]
    }
]
    
    inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=5000, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    # print(decoded)
    return decoded


def get_probability_of_true(model, processor, prompt, device):
    """
    Generates a response and calculates the probability of the model choosing '(A) True'.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        # Set output_scores to True to get the logits
        generation_output = model.generate(
            **inputs,
            max_new_tokens=5,  # We only need the next few tokens
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True # Makes the output a dict
        )

    # Get the logits for the first generated token
    first_generated_token_logits = generation_output.scores[0][0]

    # Apply softmax to get probabilities
    probabilities = F.softmax(first_generated_token_logits, dim=-1)

    # Get the token IDs for '(A)' and 'True'
    token_id_A = processor.tokenizer.encode("(A)", add_special_tokens=False)[0]
    token_id_True = processor.tokenizer.encode("True", add_special_tokens=False)[0]

    # Extract the probabilities for these tokens
    prob_A = probabilities[token_id_A].item()
    prob_True = probabilities[token_id_True].item()

    # The model might generate "(A)" or "True" as the first token.
    # We can check the probability of both and decide how to interpret.
    # For this specific prompt, the model is most likely to generate "(A)" first.
    # Therefore, we will return the probability of "(A)".

    decoded = processor.decode(generation_output.sequences[0][input_len:], skip_special_tokens=True)
    print(f"Generated text: {decoded}")

    return prob_A