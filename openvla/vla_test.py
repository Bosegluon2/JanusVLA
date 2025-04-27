# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
print("loading")
# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True, local_files_only=True,cache_dir="./cache_models")

def get_from_camera():
    # Placeholder for camera capture
    return Image.open("test.png").convert("RGB")
# Grab image input & format prompt
print("loaded")
import time
current_time = time.time()
image: Image.Image = Image.open("test.png").convert("RGB")
prompts = [
"In: What movement should the robot make to pick up the book?\nOut:",
"In: How should the robot position itself to open the door?\nOut:",
"In: What steps should the robot follow to clean the table?\nOut:",
"In: What path should the robot take to reach the charging station?\nOut:",
"In: What operations should the robot perform to assemble the toy?\nOut:"
]
import random
# while True:
#     image: Image.Image = get_from_camera()
#     prompt = prompts[random.randint(0,4)]
#     print("Prompt:",prompt)
#     inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
#     print(inputs)
#     print("Time:",time.time()-current_time)
#     current_time = time.time()

vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
     local_files_only=True,
    # low_cpu_mem_usage=True, 
    trust_remote_code=True,cache_dir="./cache_models"
).to("cuda:5", dtype=torch.bfloat16)
# Predict Action (7-DoF; un-normalize for BridgeData V2)
import time
current_time = time.time()
while True:
    inputs = processor(prompts[0], image).to("cuda:5", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(action)
    print(type(action))
    print("Time:",time.time()-current_time)
# Execute...
# robot.act(action, ...)