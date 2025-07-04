from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch.nn.functional as F
from PIL import Image
import requests
import torch

SYSTEM_PROMPT = "You are an AI assistant that helps people find information."
#SYSTEM_PROMPT = ""


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
            max_new_tokens=10,  # We only need the next few tokens
            do_sample=True,
            temperature=0.7, # A value between 0.5 and 1.0 is usually good.
            top_k=50,
            output_scores=True,
            return_dict_in_generate=True # Makes the output a dict
        )

    # Get the logits for the first generated token
    
    expected_token_id = processor.tokenizer.encode("A", add_special_tokens=False)[0]

    # Get the token ID that was *actually* generated at the second step.
    # The generated sequence is at generation_output.sequences[0].
    # The part *after* the prompt starts at index input_len.
    # The second token in that generated part is at index 'input_len + 1'.
    actual_second_generated_token_id = generation_output.sequences[0][input_len + 1].item()

    # Decode the actual token to make the printout easy to read.
    actual_second_generated_token_str = processor.decode([actual_second_generated_token_id])

    print("\n--- Verification ---")
    print(f"Expected token ID for step 2 ('(A)'): {expected_token_id}")
    print(f"Actual token ID generated at step 2 ('{actual_second_generated_token_str}'): {actual_second_generated_token_id}")
    if expected_token_id == actual_second_generated_token_id:
        print("✅ Verification successful: Logits at scores[1] are for the correct token.")
    else:
        print("❌ Verification FAILED: Logits at scores[1] are NOT for the correct token.")
        
        
    second_generated_token_logits = generation_output.scores[1][0]

    # Apply softmax to get probabilities
    stable_logits = second_generated_token_logits.to(torch.float32)
    probabilities = F.softmax(stable_logits, dim=-1)

    # Get the token IDs for '(A)' and 'True'
    token_id_A = processor.tokenizer.encode("A", add_special_tokens=False)[0]
    token_id_True = processor.tokenizer.encode("True", add_special_tokens=False)[0]
    token_id_B = processor.tokenizer.encode("B", add_special_tokens=False)[0]

    # --- Start of Debugging Block ---
    print("\n--- Full Debug Info ---")
    
    # 1. Check the Token IDs
    print(f"DEBUG: Token ID for 'A' is {token_id_A}")
    print(f"DEBUG: Token ID for 'B' is {token_id_B}")

    # 2. Inspect the Logits Tensor
    print(f"DEBUG: Logits tensor shape: {second_generated_token_logits.shape}")
    print(f"DEBUG: Logits max value: {second_generated_token_logits.max().item():.4f}")
    print(f"DEBUG: Logits min value: {second_generated_token_logits.min().item():.4f}")
    print(f"DEBUG: Logits mean value: {second_generated_token_logits.mean().item():.4f}")

    # 3. Check the specific probabilities from the full distribution
    prob_from_dist_A = probabilities[token_id_A].item()
    prob_from_dist_B = probabilities[token_id_B].item()
    print(f"DEBUG: Raw probability of token 'A' ({token_id_A}) is: {prob_from_dist_A:.4f}")
    print(f"DEBUG: Raw probability of token 'B' ({token_id_B}) is: {prob_from_dist_B:.4f}")
    
    print("-----------------------\n")

    # Extract the probabilities for these tokens
    prob_A = probabilities[token_id_A].item()
    prob_True = probabilities[token_id_True].item()
    prob_B = probabilities[token_id_B].item()

    decoded = processor.decode(generation_output.sequences[0][input_len:], skip_special_tokens=True)
    print(f"Generated text: {decoded}")
    print(f"Confidence(A)={prob_A:.2f}, Confidence(B)={prob_B:.2f}")

    return prob_A



# def get_probability_of_true(model, processor, prompt, device):
#     """
#     Calculates the probability of the model choosing '(A)' as the next token
#     by performing a direct forward pass instead of using model.generate().
#     """
#     print (f"Calculating probability for prompt: {prompt}")
#     messages = [
#         {
#             "role": "system",
#             "content": [{"type": "text", "text": SYSTEM_PROMPT}]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#             ]
#         }
#     ]
#     inputs = processor.apply_chat_template(
#         messages, add_generation_prompt=True, tokenize=True,
#         return_dict=True, return_tensors="pt"
#     ).to(device)

#     with torch.inference_mode():
#         # Perform a direct forward pass to get the logits
#         outputs = model(**inputs)
#         # Get the logits for the very last token in the input sequence
#         next_token_logits = outputs.logits[0, -1, :]

#     # Cast logits to float32 for stable softmax calculation
#     stable_logits = next_token_logits.to(torch.float32)
#     probabilities = F.softmax(stable_logits, dim=-1)

#     # Get the token ID for '(A)'
#     token_id_A = processor.tokenizer.encode("(A)", add_special_tokens=False)[0]
#     prob_A = probabilities[token_id_A].item()
    
#     # Optional: You can still print the model's top choice for debugging
#     top_token_id = torch.argmax(probabilities).item()
#     top_token_prob = probabilities[top_token_id].item()
#     top_token_str = processor.tokenizer.decode([top_token_id])
#     print(f"Model's top prediction is '{top_token_str}' with confidence {top_token_prob:.4f}")

#     return prob_A