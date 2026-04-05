import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to sys.path to allow importing from 'layers'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted

from mamba_ssm import Mamba
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # [独家创新点：动态条件跳跃残差门 (Dynamic Gated Subsumption Bypass)]
        # 它不是一个粗暴的固定常数，而是一个全连接层。让网络能根据每个交通节点（变量）的具体特征，
        # 自适应地决定“这根水管应该对这个路口开多大”。
        self.skip_gate = nn.Linear(configs.d_model, configs.d_model)
        # 初始化非常重要：-2.0 大约对应 Sigmoid 的 11% 开启率。
        # 意思是“默认主要信任 V1.0 的深层滤波，只有遇到极度刁钻的短步长突变时，才由反向传播把门动态拉开”
        torch.nn.init.constant_(self.skip_gate.bias, -2.0)
        torch.nn.init.zeros_(self.skip_gate.weight)

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # 记录未经 4 层深度平滑的原始高频特征 (Raw Embedding)
        raw_enc_out = enc_out
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # [宏观级跳跃连接 / Subsumption Architecture] 
        # 将原始高频信息强行透传到预测头，由模型自适应决定利用多少"未平滑"的原始序列
        gate = torch.sigmoid(self.skip_gate(raw_enc_out))
        enc_out = enc_out + gate * raw_enc_out

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.seq_len = 96
            self.pred_len = 96
            self.output_attention = False
            self.use_norm = True
            self.d_model = 512
            self.embed = 'timeF'
            self.freq = 'h'
            self.dropout = 0.1
            self.class_strategy = 'projection'
            self.d_state = 16
            self.d_ff = 2048
            self.activation = 'gelu'
            self.e_layers = 2

    configs = Configs()
    smamba = Model(configs=configs)
    
    print("Parameter count:", sum(p.numel() for p in smamba.parameters()))
    print(smamba)
