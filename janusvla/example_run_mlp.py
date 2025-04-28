"""
name : example_run_mlp.py
description : 使用 ActionHeadMLP 模型进行推理的示例代码。
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
from action_head import ActionHead
from action_head_mlp import ActionHeadMLP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
save_dir = 'ucsd_kitchen_dataset_split_word'
model_path = "./Janus-Pro-1B"   # 使用本地路径
assert os.path.exists(model_path), "模型路径不存在！"
def extract_first_three(tensors,need_unsqueeze=False):
    x = []
    y = []
    z = []
    for xtensor in tensors:
        
        if need_unsqueeze:
            xtensor = xtensor.cpu().numpy()[0][0]
        else:
            xtensor = xtensor.cpu().numpy()[0][0]
        x.append(xtensor[0])
        y.append(xtensor[1])
        z.append(xtensor[2])
    return np.array(x), np.array(y), np.array(z)
def maker(t,t2,name):
    for i in range(1,len(t)):
        t[i]+=(t2[0].to("cuda:5")-t[0])
    x1, y1, z1 = extract_first_three(t,True)
    x2, y2, z2 = extract_first_three(t2,True)

    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一组数据，颜色设为红色
    ax.scatter(x1, y1, z1, c='r', label='predicted')

    # 绘制第二组数据，颜色设为蓝色
    ax.scatter(x2, y2, z2, c='b', label='original')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 添加图例
    ax.legend()

    # 保存图像为 plot.png
    plt.savefig("./plot/"+str(name)+'.png')

    # 关闭图形对象
    plt.close(fig)

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_chat_processor.system_prompt = "You are a VLA that controls a robot arm."
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:5").eval()
action_head = ActionHeadMLP()
state_dict = torch.load("action_head_mlp_model.pth", map_location=torch.device('cuda:5'))
action_head.load_state_dict(state_dict)
action_head = action_head.to("cuda:5").eval()

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    loaded_dataset = SplitUCSDKitchenDataset(save_dir)
    print(f"数据集中的序列数量: {len(loaded_dataset)}")
    batch_size = 1
    dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)
    count = 0
    data_to_save = []
    sequence_counter = 0
    ss = 0
    a=1
    for sequence in dataloader:
        a+=1
        print(sequence[0][0], sequence[1].size(), sequence[2].size())
        results = []
        results2 = []
        # 初始化 history 张量
        history = torch.zeros((1, 0, 7), device=vl_gpt.device)
        cumulative_result =torch.zeros((1, 1, 7), device=vl_gpt.device)
        with torch.no_grad():  # 禁用梯度计算
            for i in range(sequence[1].size()[1]):
                if i != 0:
                    xa = sequence[2][0][i]
                    xa = xa.permute(1, 2, 0)
                    np_array = xa.numpy()
                    image = Image.fromarray(np_array)
                    #print(tensor[100][100])
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
                    #print(prepare_inputs)
                    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                    outputs = vl_gpt.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        output_hidden_states=True
                    )
                    #print(outputs.hidden_states[-1][:, -1, :])
                    hidden_states = outputs.hidden_states[-1][:, -1, :].to(vl_gpt.device)
                    sequence_1_part = sequence[1][:, :i, :].to(vl_gpt.device)

                    # 对 sequence_1_part 进行差分处理
                    if sequence_1_part.shape[1] > 1:
                        diff_sequence_1_part = sequence_1_part[:, 1:, :] - sequence_1_part[:, :-1, :]
                    else:
                        diff_sequence_1_part = sequence_1_part  # 如果只有一个时间步，差分序列就是原序列

                    # 首次计算使用差分序列，后续使用 history
                    if i == 1:
                        result = action_head(hidden_states.to(torch.float))
                    else:
                        result = action_head(hidden_states.to(torch.float))
                    cumulative_result+=result
                    # 将 result 添加到 history 后端
                    history = torch.cat((history, result.unsqueeze(0)), dim=1)

                    results.append(cumulative_result.clone())
                    results2.append(sequence[1][0][i].unsqueeze(0).unsqueeze(0))

        maker(results,results2,a)



    