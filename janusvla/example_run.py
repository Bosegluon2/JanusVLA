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

save_dir = 'ucsd_kitchen_dataset_split_word'
model_path = "./Janus-Pro-1B"   # 使用本地路径
assert os.path.exists(model_path), "模型路径不存在！"


vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_chat_processor.system_prompt = "You are a VLA that controls a robot arm."
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).to("cuda:5").eval()
action_head = ActionHead()
state_dict = torch.load("action_head_model.pth", map_location=torch.device('cuda:5'))
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
    for sequence in dataloader:

        print(sequence[0][0], sequence[1].size(), sequence[2].size())
        results = []
        results2 = []
        # 初始化 history 张量
        history = torch.zeros((1, 0, 7), device=vl_gpt.device)
        cumulative_result =torch.zeros((1, 1, 7), device=vl_gpt.device)
        with torch.no_grad():  # 禁用梯度计算
            for i in range(sequence[1].size()[1]):
                if i != 0:
                    tensor = sequence[2][0][i]
                    tensor = tensor.permute(1, 2, 0)
                    np_array = tensor.numpy()
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
                        result = action_head(hidden_states, diff_sequence_1_part)
                    else:
                        result = action_head(hidden_states, history)
                    cumulative_result+=result
                    # 将 result 添加到 history 后端
                    history = torch.cat((history, result.unsqueeze(0)), dim=1)

                    results.append(cumulative_result.clone())
                    results2.append(sequence[1][0][i].unsqueeze(0).unsqueeze(0))

        break  # 只用第一个作为测试
    print(results)
    print(results2)