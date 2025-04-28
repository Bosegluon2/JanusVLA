"""
name : action_head_mlp.py
description : ActionHeadMLP 模块。
author : 汪子策
date : 2025-4-14
version : 1.0
license : ARR
Copyright (c) 2025 汪子策. All rights reserved.
"""
import torch
from torch import nn
class ActionHeadMLP(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(ActionHeadMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7)
        )

    def forward(self, x_2048):

        output = self.fc(x_2048)
        return output

# action_head=ActionHead().to(vl_gpt.device)