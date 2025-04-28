"""
name : build_dataset.py
description : Preprocess the dataset. 
note: 这段代码可读性很差，这是因为汪子策故意的。
tips: 这段代码需要两个不同的环境来运行。
author : 汪子策
date : 2025-4-14
version : 1.0
license : ARR
Copyright (c) 2025 汪子策. All rights reserved.
"""
# -*- coding: utf-8 -*-
# import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# import tiktoken
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # 加载数据集
# (ds_train,), ds_info = tfds.load(
#     'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
#     download=True,
#     data_dir='./datasets',
#     split=['train'],
#     shuffle_files=True,
#     with_info=True
# )
# num_train_examples = ds_info.splits['train'].num_examples
# print(f"训练数据集的样本数量: {num_train_examples}")
# # 自定义 PyTorch 数据集类
# class UCSDKitchenDataset(Dataset):
#     def __init__(self, tf_dataset):
#         # self.data = []
#         # count = 0
#         # for example in tf_dataset:
#         #     count += 1
#         #     if count % 100 == 0:
#         #         print(count)
#         #     sequence = []
            
#         #     steps = example['steps']

#         #     for step in steps:
#         #         #print(step['language_instruction'].numpy().decode('utf-8'))
#         #         language_instruction = torch.tensor(encoding.encode(step['language_instruction'].numpy().decode('utf-8')))
#         #         state = torch.from_numpy(step['observation']['state'].numpy())
#         #         image = torch.from_numpy(step['observation']['image'].numpy()).permute(2, 0, 1)
#         #         sequence.append((language_instruction, state, image))
#         #     self.data.append(sequence)
#         # print(count)
#         self.data = []
#         count = 0
#         for example in tf_dataset:
#             count += 1
#             if count % 100 == 0:
#                 print(count)
#             # 初始化三个序列列表
#             instruct_list = []
#             state_list = []
#             image_list = []

#             steps = example['steps']

#             for step in steps:
#                 # 处理语言指令
#                 language_instruction = step['language_instruction'].numpy().decode('utf-8')#torch.tensor(encoding.encode(step['language_instruction'].numpy().decode('utf-8')))
#                 instruct_list.append(language_instruction)

#                 # 处理状态
#                 state = torch.from_numpy(step['observation']['state'].numpy())
#                 state_list.append(state)

#                 # 处理图像
#                 image = torch.from_numpy(step['observation']['image'].numpy()).permute(2, 0, 1)
#                 image_list.append(image)

#             # 将列表转换为 tensor
#             #instruct_sequence = torch.stack(instruct_list) if instruct_list else torch.tensor([])
#             state_sequence = torch.stack(state_list) if state_list else torch.tensor([])
#             image_sequence = torch.stack(image_list) if image_list else torch.tensor([])

#             # 构建三元组并添加到 self.data 中
#             self.data.append((language_instruction, state_sequence, image_sequence))

#         print(count)


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # 创建 PyTorch 数据集
# pytorch_dataset = UCSDKitchenDataset(ds_train)

# # 每两百个 sequence 分一个文件保存
# save_dir = 'ucsd_kitchen_dataset_split_word'
# os.makedirs(save_dir, exist_ok=True)
# chunk_size = 200
# for i in range(0, len(pytorch_dataset), chunk_size):
#     end_index = min(i + chunk_size, len(pytorch_dataset))
#     chunk = pytorch_dataset.data[i:end_index]
#     save_path = os.path.join(save_dir, f'chunk_{i // chunk_size}.pt')
#     torch.save(chunk, save_path)
# print(f"数据集已按每 {chunk_size} 个 sequence 拆分保存到 {save_dir}")

# save_dir = 'ucsd_kitchen_dataset_split_word'
# # 从多个文件加载数据集
# class SplitUCSDKitchenDataset(Dataset):
#     def __init__(self, save_dir):
#         # self.save_dir = save_dir
#         # self.file_list = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.pt')]
#         # self.data = []
#         # for file_path in self.file_list:
#         #     self.data.extend(torch.load(file_path))
#         self.save_dir = save_dir
#         self.file_list = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.pt')]
#         self.data = []
#         for file_path in self.file_list:
#             sequences = torch.load(file_path,weights_only=True)
#             for sequence in sequences:
#                 if len(sequence) == 3:
#                     instruct_sequence, state_sequence, image_sequence = sequence
#                     # 确保元素是 tensor 类型，如果不是则进行转换
#                     # if not isinstance(instruct_sequence, torch.Tensor):
#                     #    instruct_sequence = torch.tensor(instruct_sequence)
#                     if not isinstance(state_sequence, torch.Tensor):
#                         state_sequence = torch.tensor(state_sequence)
#                     if not isinstance(image_sequence, torch.Tensor):
#                         image_sequence = torch.tensor(image_sequence)
#                     self.data.append((instruct_sequence, state_sequence, image_sequence))
#                 else:
#                     print(f"Unexpected sequence length: {len(sequence)}")
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# if __name__ == '__main__':
#     torch.set_printoptions(sci_mode=False)
#     import matplotlib
#     import matplotlib.pyplot as plt


# # 创建基于拆分数据的数据集和数据加载器
#     loaded_dataset = SplitUCSDKitchenDataset(save_dir)
#     print(f"数据集中的序列数量: {len(loaded_dataset)}")
#     batch_size = 1
#     dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)
#     count = 0
#     for sequence in dataloader:
        
#         print(sequence[0][0], sequence[1].shape, sequence[2].shape)

#         from PIL import Image
#         from torchvision.transforms import ToPILImage
        
#         to_pil = ToPILImage()

#         tensor=sequence[2][0][1]
#         print(tensor)

#         tensor = tensor.permute(1, 2, 0)
        

#         np_array = tensor.numpy()
#         print(np_array)
#         image = Image.fromarray(np_array)
        
#         image.save("test.png")
