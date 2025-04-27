import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys

# 加载图像并进行预处理
image = Image.open("test.png").convert("RGB")

# 定义通用的图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image).unsqueeze(0).to('cuda')

# 使用 DINOv2 模型
print("使用 DINOv2 模型:")
model_dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model_dinov2.eval().cuda()

with torch.no_grad():
    output_dinov2 = model_dinov2(input_tensor)
print("DINOv2 输出形状:", output_dinov2.shape)
for i in range(5):
    print(f"第 {i+1} 个输出:", output_dinov2[0, i].cpu().numpy())
# 使用 ConvNeXt 模型
print("\n使用 ConvNeXt 模型:")
model_convnext = models.convnext_tiny(pretrained=True)
model_convnext.eval().cuda()

with torch.no_grad():
    output_convnext = model_convnext(input_tensor)
print("ConvNeXt 输出形状:", output_convnext.shape)

import json
with open('imagenet-simple-labels.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)


top3_scores, top3_indices = torch.topk(output_convnext, 3, dim=1)
top3_indices = top3_indices.squeeze().tolist()
top3_scores = top3_scores.squeeze().tolist()

# 输出前三个预测结果
print("前三个预测结果：")
for i in range(3):
    class_name = class_names[top3_indices[i]]
    score = top3_scores[i]
    print(f"{i + 1}. 类别: {class_name}, 分数: {score:.4f}")
# 使用 SigLIP 模型（模拟使用，实际需从官方仓库获取代码）
print("\n使用 SigLIP 模型:")
try:
    # 假设已将 SigLIP 代码路径添加到系统路径
    sys.path.append('path/to/siglip_repo')  # 替换为实际的 SigLIP 仓库路径
    from siglip_model import SigLIPModel  # 假设的 SigLIP 模型类
    model_siglip = SigLIPModel()
    model_siglip.eval()

    with torch.no_grad():
        output_siglip = model_siglip(input_tensor)
    print("SigLIP 输出形状:", output_siglip.shape)
except ImportError:
    print("SigLIP 模型导入失败，请检查路径和依赖。实际使用时需要从官方仓库获取代码并正确配置。")