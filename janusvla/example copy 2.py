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
class ActionHead(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(ActionHead, self).__init__()
        self.d_model = d_model

        # 嵌入层将 7 维输入映射到 d_model 维
        self.embedding = nn.Linear(7, d_model)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
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

action_head=ActionHead().to(vl_gpt.device)
if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    loaded_dataset = SplitUCSDKitchenDataset(save_dir)
    print(f"数据集中的序列数量: {len(loaded_dataset)}")
    batch_size = 1
    dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)
    count = 0
    # for sequence in dataloader:
        
    #     print(sequence[0][0],sequence[1].size(),sequence[2].size())
    #     for i in range(sequence[1].size()[1]):
            
    #         tensor=sequence[2][0][i]
    #         tensor = tensor.permute(1, 2, 0)
            

    #         np_array = tensor.numpy()
    #         image = Image.fromarray(np_array)
    #         conversation = [
    #             {
    #                 "role": "<|User|>",
    #                 "content": f"<image_placeholder>\n{sequence[0][0]}",
    #                 "images": ["camera.png"],
    #             },
    #             {"role": "<|Assistant|>", "content": ""},
    #         ]

    #         pil_images = [image]
    #         prepare_inputs = vl_chat_processor(
    #             conversations=conversation, images=pil_images, force_batchify=True
    #         ).to(vl_gpt.device)
            
    #         inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            

    #         outputs = vl_gpt.language_model(
    #                 inputs_embeds=inputs_embeds,
    #                 attention_mask=prepare_inputs.attention_mask,
    #                 output_hidden_states=True
    #         )
            
    #         outputs2=action_head(outputs.hidden_states[0][:, 0, :].to(vl_gpt.device),sequence[1][:i+1].to(vl_gpt.device))
    #         print(outputs2.size())
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
            sequence_1_part = sequence[1][:i + 1].to(vl_gpt.device)
            data_to_save.append((hidden_states, sequence_1_part))

            # outputs2 = action_head(hidden_states, sequence_1_part)
            print(hidden_states.size())
            print(sequence_1_part.size())
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
 

