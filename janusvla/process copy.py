
"""
name : process copy.py
description : 处理数据集，提取隐藏状态和历史步骤，并保存为 PyTorch 张量文件。
author : 汪子策
date : 2025-4-14
version : 1.0
license : All rights reserved.
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
vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:5").eval()

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
    last_tensor=torch.tensor([[1,1,1,1,1,1]])
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
            hidden_states = outputs.hidden_states[0][:, 0, :].to(vl_gpt.device)
            history_step = sequence[1][:i + 1].to(vl_gpt.device)
            data_to_save.append((hidden_states, history_step))

            # outputs2 = action_head(hidden_states, history_step)
            print(hidden_states.size())
            print(history_step.size())
            if last_tensor.size()==hidden_states.size():
                print("Cosine similarity: ", nn.functional.cosine_similarity(last_tensor,hidden_states))
            last_tensor=hidden_states
        sequence_counter += 1
        if sequence_counter % save_interval == 0:
            # 保存数据
            torch.save(data_to_save, f'saved_data_{sequence_counter // save_interval}.pt')
            data_to_save = []  

    if data_to_save:
        torch.save(data_to_save, f'saved_data_{(sequence_counter // save_interval) + 1}.pt')
 

