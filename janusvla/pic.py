"""
name : pic.py
description : modified from origin janus repository, to test the action head.
author : 汪子策
date : 2025-4-14
version : 1.0
license : MIT License
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import os
 
# specify the path to the model
# model_path = "deepseek-ai/Janus-Pro-1B"  # 禁用此远程路径
model_path = "./Janus-Pro-1B"   # 使用本地路径
assert os.path.exists(model_path), "模型路径不存在！"
 
 
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
 
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# # 准备对话和图像
question = '图片中，都有什么？' #可用中英文
image_path = './images/predict.png' #官方的doge梗图
assert os.path.exists(image_path), "图像路径不存在！"
 
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image_path],
    },
    {"role": "<|Assistant|>", "content": ""},
]
 

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)
 
# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)




# 在模型初始化后添加动作头
action_head = ActionHead().to(vl_gpt.device).bfloat16()


from peft import LoraConfig, get_peft_model

# 配置低秩适配器
lora_config = LoraConfig(
    r=8,                  # 秩
    lora_alpha=32,        # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 注入低秩更新的模块
    lora_dropout=0.1,
)

# 应用至基础模型

peft_model = get_peft_model(vl_gpt, lora_config)


def get_action_features(model, prepare_inputs):
    # 获取多模态融合特征
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=prepare_inputs.inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            output_hidden_states=True
        )
    # 取最后一个隐藏层的CLS token特征
    last_hidden = outputs.hidden_states[-1][:, 0, :]  # [batch, hidden_size]
    return last_hidden

# 替换原有生成循环
import time
current_time = time.time()
while True:
    # 获取多模态特征
    with torch.no_grad():
        hidden_features = get_action_features(peft_model, prepare_inputs)
    
    # 预测动作参数
    action = action_head(hidden_features)
    
    # 反归一化到实际坐标范围（假设工作空间为0~1）
    action = (action + 1) / 2
    print(f"Predicted Action: {action.detach().cpu().numpy()}")
    print(time.time()-current_time)
    current_time = time.time()
