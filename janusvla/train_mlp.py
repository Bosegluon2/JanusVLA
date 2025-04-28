"""
name : train_mlp.py
description : Trains the ActionHeadMLP model using data from saved_data_1.pt to saved_data_46.pt.
author : 汪子策
date : 2025-4-14
version : 1.0
license : All rights reserved.
Copyright (c) 2025 汪子策. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from action_head import ActionHead
from action_head_mlp import ActionHeadMLP
# 定义保存文件的目录
save_dir = 'test'
# 初始化模型
model = ActionHeadMLP().to("cuda:5")
# model.load_state_dict(torch.load("action_head_10.pth",weights_only=True))
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练轮数
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    total_cnt = 0
    # 分批次加载数据集，每次加载两份
    for batch_start in range(1, 47):
        loaded_data = []
        for i in range(batch_start, min(batch_start + 1, 47)):
            try:
                # 构建文件路径
                file_path = f'{save_dir}/saved_data_{i}.pt'
                # 加载文件到 GPU
                data = torch.load(file_path, map_location=torch.device('cuda:5'))
                # 将加载的数据添加到列表中
                loaded_data.extend(data)
            except FileNotFoundError:
                print(f"文件 {file_path} 未找到。")
            except Exception as e:
                print(f"加载文件 {file_path} 时发生错误: {e}")

        cnt = 0
        for hidden_states, sequence_1_part in loaded_data:
            # 跳过 n = 1 的情况
            if sequence_1_part.size(1) == 1 or sequence_1_part.size(1) == 2:
                continue
            cnt += 1
            total_cnt += 1
            if total_cnt % 100 == 0:
                print(total_cnt)

            # 拆分训练数据和标签
            inputs = sequence_1_part[:, :-1, :]


            # 计算差分序列
            diff_sequence = inputs[:, 1:, :] - inputs[:, :-1, :]
            # print(inputs.shape)
            # print(diff_sequence.shape)
            labels = sequence_1_part[:, -1:, :]-sequence_1_part[:, -2:-1, :]

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(hidden_states.to(torch.float))

            # 计算损失
            loss = criterion(outputs.unsqueeze(0), labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # 打印每个 epoch 的损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / total_cnt}')

print('训练完成！')

# 保存模型参数
model_save_path = 'action_head_mlp_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"模型参数已保存到 {model_save_path}")