import torch
import torch.nn as nn
import math

# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Vision Transformer 图像编码器
class ViTImageEncoder(nn.Module):
    def __init__(self, image_size=64, patch_size=16, d_model=128, num_heads=8, num_layers=2):
        super(ViTImageEncoder, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_patches + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, images):
        # images: [batch_size, seq_len, 3, image_size, image_size]
        batch_size, seq_len, _, _, _ = images.shape
        images = images.view(batch_size * seq_len, 3, images.size(-2), images.size(-1))
        patches = self.patch_embedding(images)
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size * seq_len, num_patches, d_model]
        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [num_patches + 1, batch_size * seq_len, d_model]
        output = self.transformer_encoder(x)
        cls_output = output[0]
        cls_output = cls_output.view(batch_size, seq_len, -1)
        return cls_output

# 语义理解 Transformer 模型
class SemanticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=2, max_seq_len=100, image_size=64):
        super(SemanticTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.image_encoder = ViTImageEncoder(image_size, d_model=d_model, num_heads=nhead, num_layers=num_layers)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 7)

    def forward(self, src_text, src_images):
        # 文本输入处理
        text_embedded = self.embedding(src_text)
        text_embedded = self.positional_encoding(text_embedded)

        # 图像输入处理
        image_encoded = self.image_encoder(src_images)
        image_encoded = image_encoded.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        image_encoded = self.positional_encoding(image_encoded)

        # 合并文本和图像输入
        combined_input = torch.cat([text_embedded, image_encoded], dim=0)

        # Transformer 编码器
        output = self.transformer_encoder(combined_input)
        # 取序列的第一个位置的输出
        output = output[0]
        print(output.shape)
        # 通过全连接层映射到 7 维空间
        output = self.fc(output)
        return output


# 示例使用
vocab_size = 1000  # 词汇表大小
model = SemanticTransformer(vocab_size)

# 模拟文本输入，假设输入序列长度为 10
input_text = torch.randint(0, vocab_size, (10, 1))
# 模拟图像输入，假设图像序列长度为 5，图像大小为 64x64，通道数为 3
input_images = torch.randn(1, 5, 3, 64, 64)

output = model(input_text, input_images)
print("Output shape:", output.shape)