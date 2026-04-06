import os

with open("/data/home/hzl/code/S-D-Mamba/model/S_Mamba_Patch.py", "r") as f:
    content = f.read()

import re

# We will regex replace from "class LocalGlobalCrossChannelMixing" up to "class Model"
pattern = r"class LocalGlobalCrossChannelMixing.*?class Model"

replacement = """class FFN_Patch(nn.Module):
    def __init__(self, d_model=128, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expand)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * expand, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x + residual


class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, d_state=32, expand=2, dropout=0.1):                                            
        super().__init__()
        self.bi_idmb = Bi_IDMB_Patch(d_model, d_state, expand, dropout)
        self.ffn = FFN_Patch(d_model, expand, dropout)

    def forward(self, x, B, N, P_num):
        x = self.bi_idmb(x)
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, e_layers=2, d_model=128, d_state=32, expand=2, dropout=0.1):                                
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_state, expand, dropout)                                                                  
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, B, N, P_num):
        for layer in self.layers:
            x = layer(x, B, N, P_num)
        x = self.norm(x)
        return x


class Model"""

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("/data/home/hzl/code/S-D-Mamba/model/S_Mamba_Patch.py", "w") as f:
    f.write(new_content)

