import torch
from torch import nn
class ActionHead(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(ActionHead, self).__init__()
        self.d_model = d_model

        # 嵌入层将 7 维输入映射到 d_model 维
        self.embedding = nn.Linear(7, d_model)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层将 2048 维特征和 Transformer 输出拼接后映射到最终输出
        self.fc = nn.Sequential(
            nn.Linear(2048 + d_model, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7)
        )

    def forward(self, x_2048, x_seq):

        embedded = self.embedding(x_seq)  
        embedded = embedded.permute(1, 0, 2)  


        transformer_output = self.transformer_encoder(embedded)  

        transformer_last_output = transformer_output[-1, :, :]  


        combined = torch.cat((x_2048, transformer_last_output), dim=1)

        output = self.fc(combined)
        return output

# action_head=ActionHead().to(vl_gpt.device)