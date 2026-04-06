import torch
import torch.nn as nn
from model.S_Mamba import Model

class Config:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.output_attention = False
        self.use_norm = 1
        self.d_model = 128
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1
        self.class_strategy = 'projection'
        self.d_state = 16
        self.d_conv = 4
        self.expand = 2
        self.e_layers = 1
        self.d_ff = 128
        self.activation = 'gelu'

configs = Config()
model = Model(configs)
print(model)
