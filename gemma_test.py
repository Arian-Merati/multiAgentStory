# # pip install accelerate

# from transformers import AutoProcessor, Gemma3ForConditionalGeneration
# from PIL import Image
# import requests
# import torch

# model_id = "google/gemma-3-4b-it"

# model = Gemma3ForConditionalGeneration.from_pretrained(
#     model_id, device_map="auto"
# ).eval()

# processor = AutoProcessor.from_pretrained(model_id)

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Write a short story about a robot."},
#         ]
#     }
# ]

# inputs = processor.apply_chat_template(
#     messages, add_generation_prompt=True, tokenize=True,
#     return_dict=True, return_tensors="pt"
# ).to(model.device, dtype=torch.bfloat16)

# input_len = inputs["input_ids"].shape[-1]

# with torch.inference_mode():
#     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]

# decoded = processor.decode(generation, skip_special_tokens=True)
# print(decoded)

# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

# 1. Explicitly define the device
device = "mps"
model_id = "google/gemma-3-12b-it"

# 2. Load the model and send it to the specified device.
#    It's also a good practice to specify the torch_dtype here.
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
).to(device).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Write a short and coherent story about Superman that incorporates the answers to the following 5 questions: What was Warren Beatty's first movie? Which future Hollywood star got her break as Wonder Girl, Wonder Woman's sister Drusilla? Mickey Braddock in the 50s series Circus Boy found fame with which surname in which pop band? Which war veteran was Director of News & Special Events for ABC before find fame as a TV cop? What US sitcom was the first exported to Britain?"},
            #{"type": "text", "text": "Answer the following five questions first: What was Warren Beatty's first movie? Which future Hollywood star got her break as Wonder Girl, Wonder Woman's sister Drusilla? Mickey Braddock in the 50s series Circus Boy found fame with which surname in which pop band? Which war veteran was Director of News & Special Events for ABC before find fame as a TV cop? What US sitcom was the first exported to Britain? Now incorporate them into a short and coherent story about Superman."},z
        ]
    }
]

# 3. Ensure the inputs are sent to the same device as the model.
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)