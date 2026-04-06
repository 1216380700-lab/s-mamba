import torch
import torch.nn as nn
import sys
import os

from einops import rearrange
from mamba_ssm import Mamba

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from layers.Mamba_EncDec import AdaptiveFourierFilterBlock

class Bi_IDMB_Patch(nn.Module):
    """
    专为Patch时序维度设计的双向双尺度Mamba块
    严格保证因果性，无未来信息泄露，解决方差爆炸问题
    """
    def __init__(self, d_model=128, d_state=32, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        
        self.mamba_local = Mamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=4,  # 小核，抓局部
        )
        self.mamba_global = Mamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=8,  # 大核，抓全局
        )
        
        self.gate_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        
        feat_local = self.mamba_local(x_norm)
        feat_global = self.mamba_global(x_norm)
        
        x_reversed = torch.flip(x_norm, dims=[1])
        feat_local_rev = torch.flip(self.mamba_local(x_reversed), dims=[1])
        feat_global_rev = torch.flip(self.mamba_global(x_reversed), dims=[1])
        
        feat_forward = feat_local + feat_global
        feat_reverse = feat_local_rev + feat_global_rev
        
        gate = self.gate_fusion(torch.cat([feat_forward, feat_reverse], dim=-1))
        feat_fused = gate * feat_forward + (1 - gate) * feat_reverse
        feat_fused = self.out_proj(feat_fused)

        return self.dropout(feat_fused)


class FFN_Patch(nn.Module):
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
        self.norm1 = nn.LayerNorm(d_model)
        self.bi_idmb = Bi_IDMB_Patch(d_model, d_state, expand, dropout)
        self.ffn = FFN_Patch(d_model, expand, dropout)

    def forward(self, x, B, N, P_num):
        x = x + self.bi_idmb(self.norm1(x))
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


class Model(nn.Module):
    """
    Patching + CI + Bi-IDMB + Affirm Layer Style
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.patch_size = min(getattr(configs, 'patch_len', 24), self.seq_len)
        self.stride = getattr(configs, 'stride', self.patch_size // 2)
        self.stride = max(1, min(self.stride, self.patch_size))
        self.d_model = configs.d_model
        self.d_state = getattr(configs, 'd_state', 16)
        self.expand = getattr(configs, 'expand', 1)

        self.num_patches = int((self.seq_len - self.patch_size) / self.stride + 1)

        self.use_norm = configs.use_norm
        self.enc_in = configs.enc_in

        self.frontend_affb = AdaptiveFourierFilterBlock(self.seq_len)
        
        self.patch_embedding = nn.Linear(self.patch_size, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        
        self.encoder = Encoder(
            e_layers=configs.e_layers,
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            dropout=configs.dropout
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        self.projector = nn.Linear(self.num_patches * self.d_model, self.pred_len)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape
        
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = self.frontend_affb(x_enc)

        # CI独立 (Channel Independence) 维度重排
        x_enc = rearrange(x_enc, 'b l n -> b n l')
        
        # 50% 重叠连续 Patching
        x_patched = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b n p_n p_s -> (b n) p_n p_s')
        
        # Embedding
        enc_out = self.patch_embedding(x_patched)
        enc_out = enc_out + self.position_embedding
        
        # Affirm 式编码 (AFFB + Bi-IDMB + FFN)
        enc_out = self.encoder(enc_out, B, N, self.num_patches)
        
        # Projection
        enc_out = self.flatten(enc_out)
        dec_out = self.projector(enc_out)
        dec_out = rearrange(dec_out, '(b n) l -> b l n', b=B, n=N)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]