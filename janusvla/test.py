"""
name : test.py
description : 处理数据集，提取隐藏状态和历史步骤，并保存为 PyTorch 张量文件。
author : 汪子策
date : 2025-4-14
version : 1.0
license : ARR
Copyright (c) 2025 汪子策. All rights reserved.
"""
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import os
from build_dataset import SplitUCSDKitchenDataset
save_dir = 'ucsd_kitchen_dataset_split_word'
model_path = "./Janus-Pro-1B"   # 使用本地路径
assert os.path.exists(model_path), "模型路径不存在！"
 
 
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_chat_processor.system_prompt="You are a VLA that controls a robot arm."
tokenizer = vl_chat_processor.tokenizer
 
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:5").eval()#cuda().eval()

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    loaded_dataset = SplitUCSDKitchenDataset(save_dir)
    print(f"数据集中的序列数量: {len(loaded_dataset)}")
    batch_size = 1
    dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)
    count = 0
    data_to_save = []
    sequence_counter = 0
    save_interval = 200
    ss=0
    for sequence in dataloader:
        print(sequence[0][0], sequence[1].size(), sequence[2].size())
        for i in range(sequence[1].size()[1]):
            tensor = sequence[2][0][i]
            tensor = tensor.permute(1, 2, 0)
            np_array = tensor.numpy()
            image = Image.fromarray(np_array)
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{sequence[0][0]}",
                    "images": ["camera.png"],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = [image]
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = vl_gpt.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1][:, 0, :].to(vl_gpt.device)
            sequence_1_part = sequence[1][:i + 1].to(vl_gpt.device)
            data_to_save.append((hidden_states, sequence_1_part))


        sequence_counter += 1
        if sequence_counter % save_interval == 0:
        
            torch.save(data_to_save, f'test/saved_data_{ss}.pt')
            data_to_save = []  
            ss+=1

    
    if data_to_save:
        torch.save(data_to_save, f'test/saved_data_{(sequence_counter // save_interval) + 1}.pt')
 

