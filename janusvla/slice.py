import torch

# 假设 sequence[1] 是一个固定形状的张量
sequence = [None, torch.randn(1, 100, 7)]  # 这里假设 sequence[1] 的形状为 [1, 100, 7]

for i in range(1, 10):
    sequence_1_part = sequence[1][:, :i, :].to('cpu')  # 这里将其移动到 CPU 上，方便打印
    print(f"i: {i}, sequence_1_part.shape: {sequence_1_part.shape}")