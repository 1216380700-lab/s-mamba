
import torch
import torch.nn as nn

class AdaptiveFourierFilterBlock(nn.Module):
    """
    Phase 1: Time-domain Adaptive Fourier Filter Block (frontend).
    Applied BEFORE variable inversion, acting directly on the true Temporal dimension (Length).
    Input: [B, L, N]
    """
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        d_fft = seq_len // 2 + 1
        
        # Learnable adaptive cutoff frequencies (normalized to 0~1)
        self.low_pass_cut_freq_param = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.high_pass_cut_freq_param = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Learnable complex weights for global, low-pass, and high-pass branches
        # [1, d_fft, 1, 2] -> broadcasts over Batch and Variates
        self.complex_weight_global = nn.Parameter(torch.randn(1, d_fft, 1, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_low = nn.Parameter(torch.randn(1, d_fft, 1, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(1, d_fft, 1, 2, dtype=torch.float32) * 0.02)

    def apply_complex_weight(self, x_fft, weight):
        # x_fft: [B, d_fft, N] (complex)
        w = torch.view_as_complex(weight)
        # 根据豆包报告：整体执行 element-wise 复数乘法，而非分别激活实部虚部从而破坏相位信息
        x_weighted = x_fft * w
        return x_weighted

    def forward(self, x_in):
        # True time dimension is at dim=1
        x_fft = torch.fft.rfft(x_in, dim=1, norm='ortho')  # [B, L//2+1, N]
        B, D_fft, N = x_fft.shape
        
        # Sigmoid-based differentiable masks for adaptive frequency band separation
        freq = torch.linspace(0, 1, D_fft).view(1, D_fft, 1).to(x_in.device)
        tau = 10.0 # temperature for steepness
        low_mask = torch.sigmoid((self.low_pass_cut_freq_param - freq) * tau)
        high_mask = torch.sigmoid((freq - self.high_pass_cut_freq_param) * tau)
        
        x_low_pass = x_fft * low_mask
        x_high_pass = x_fft * high_mask

        # Summation over three branches (Global, Low-pass, High-pass)
        x_out_fft = (
            self.apply_complex_weight(x_fft, self.complex_weight_global) + 
            self.apply_complex_weight(x_low_pass, self.complex_weight_low) + 
            self.apply_complex_weight(x_high_pass, self.complex_weight_high)
        )
        
        # Inverse FFT back to original time domain
        x_out = torch.fft.irfft(x_out_fft, n=self.seq_len, dim=1, norm='ortho')
        
        # 加上残差连接，防止频域滤波在初期把高频有效信息全部截断导致梯度消失
        return x_in + x_out


class EncoderLayer(nn.Module):
    """
    Phase 2: Inverted Variable Encoding Layer.
    Pure dual-Mamba + FFN architecture operating on [B, N, E].
    """
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # VC Extractors: Bidirectional Mambas for variable correlation
        self.attention = attention
        self.attention_r = attention_r
        
        # TD/Feature Extractors: FFN
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 1. VC (Variable Correlation) encoding using Bi-directional Mamba
        # forward pass + backward pass
        x_fwd = self.attention(x)
        x_bwd = self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        new_x = x_fwd + x_bwd
        
        x = x + self.dropout(new_x)
        
        y = self.norm1(x)
        
        # 2. TD (Time-Dependent representation scaling) using FFN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), 1


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
