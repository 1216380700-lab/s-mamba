import torch.nn as nn
import torch


def complex_relu(x):
    real = torch.relu(x.real)
    imag = torch.relu(x.imag)
    return torch.complex(real, imag)


class LearnableFilterLayer(nn.Module):
    def __init__(self, dim):
        super(LearnableFilterLayer, self).__init__()
        self.complex_weight_1 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_2 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x_fft):
        weight_1 = torch.view_as_complex(self.complex_weight_1)
        weight_2 = torch.view_as_complex(self.complex_weight_2)
        x_weighted = x_fft * weight_1
        x_weighted = complex_relu(x_weighted)
        x_weighted = x_weighted * weight_2
        return x_weighted


class AdaptiveFourierFilterBlock(nn.Module):
    """AFFB adapted for S-Mamba: apply FFT on feature dimension (last dim)."""

    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        # [修复] rfft 变换后，特征维度的长度会变为 dim // 2 + 1
        # 所以必须用这个长度初始化滤波器，否则会在前向传播时发生 shape 不匹配的崩溃！
        d_fft = dim // 2 + 1
        self.learnable_filter_layer_1 = LearnableFilterLayer(d_fft)
        self.learnable_filter_layer_2 = LearnableFilterLayer(d_fft)
        self.learnable_filter_layer_3 = LearnableFilterLayer(d_fft)

        self.low_pass_cut_freq_param = nn.Parameter(torch.tensor(dim / 2.0 - 0.5, dtype=torch.float32))
        self.high_pass_cut_freq_param = nn.Parameter(torch.tensor(dim / 4.0 - 0.5, dtype=torch.float32))

    def adaptive_freq_pass(self, x_fft, flag="high"):
        # Inherit S-Mamba variable sequence V
        _, _, c = x_fft.shape
        D_original = (c - 1) * 2
        freq = torch.fft.rfftfreq(D_original, d=1.0).to(x_fft.device)
        if flag == "high":
            freq_mask = torch.abs(freq) >= self.high_pass_cut_freq_param.to(x_fft.device)
        else:
            freq_mask = torch.abs(freq) <= self.low_pass_cut_freq_param.to(x_fft.device)
        return x_fft * freq_mask.view(1, 1, -1)

    def forward(self, x_in):
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # In inverted time-series modeling x is [B, N, D], and AFFB should mix feature dimension D.
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')

        if self.adaptive_filter:
            x_low_pass = self.adaptive_freq_pass(x_fft, flag="low")
            x_high_pass = self.adaptive_freq_pass(x_fft, flag="high")
        else:
            x_low_pass = x_fft
            x_high_pass = x_fft

        x_weighted = (
            self.learnable_filter_layer_1(x_fft)
            + self.learnable_filter_layer_2(x_low_pass)
            + self.learnable_filter_layer_3(x_high_pass)
        )
        # Use irfft back to D real dimension
        x = torch.fft.irfft(x_weighted, n=x.shape[-1], dim=-1, norm='ortho')

        return x.to(dtype)


class Bi_IDMB_Block(nn.Module):
    """
    Stationary Bi-IDMB Block (Global Adaptive Gating & Distribution Align)
    Resolves Variance Explosion & FFT requirement clash by avoiding cross-multiplication.
    """
    def __init__(self, dim, drop=0.):
        super().__init__()
        # Global Adaptive Gating
        self.silu = nn.SiLU()
        self.gate_proj = nn.Linear(dim, 2)
        
        self.out_proj = nn.Conv1d(dim, dim, 1)
        
        # Distribution alignment for downstream AFFB (FFT stability)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x, mamba_fwd, mamba_bwd):
        # Branch 1: Forward Mamba
        x1 = mamba_fwd(x)
        
        # Branch 2: Backward Mamba
        x2 = mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        
        # Softmax Gating Fusion
        # Instead of cross-multiplying, learn to weight them adaptively
        gates = self.gate_proj(self.silu(x1 + x2)) # (B, V, 2)
        gates = torch.softmax(gates, dim=-1) # (B, V, 2)
        weight_1 = gates[..., 0:1] # (B, V, 1)
        weight_2 = gates[..., 1:2] # (B, V, 1)
        
        # Weighted additive sum, interacting with the original input's activation state
        x_fused = (x1 * weight_1) + (x2 * weight_2)
        x_act = self.silu(x_fused) * self.silu(x)
        
        # Transpose for 1x1 Conv channel mixing
        out = x_act.transpose(2, 1) # (B, C, N)
        out = self.out_proj(out)
        out = out.transpose(1, 2) # (B, N, C) <=> (B, V, D)
        
        # Force stationary distribution before routing back to residual / AFFB
        out = self.out_norm(out)
        
        return out


class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.attention_r = attention_r
        d_ff = d_ff or 4 * d_model

        # Bi-IDMB Cross-gating module
        self.bi_idmb = Bi_IDMB_Block(d_model, drop=dropout)

        # Time-domain FFN (Pure FFN for isolation test)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

        # Frequency-domain AFFB
        self.affb = AdaptiveFourierFilterBlock(d_model, adaptive_filter=True)
        
        # Adaptive Gated Fusion
        self.w_gate = nn.Linear(d_model, d_model)
        torch.nn.init.constant_(self.w_gate.bias, 3.5)  # Init bias towards FFN (high reliance on Time-domain initially)
        torch.nn.init.zeros_(self.w_gate.weight)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Trainable variable scaling 
        self.v_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        V = x.size(1)  # Variable dimension (Sequence length in S-Mamba inverted view)
        
        # Replaced naive addition with Bi-IDMB Cross-Gating
        new_x = self.bi_idmb(x, self.attention, self.attention_r)
        attn = 1

        # Use generic initialized scaling factor instead of hardcoded magic number
        x = x + self.v_weight * new_x
        
        y = self.norm1(x)  # DON'T overwrite x with y, pre-LN architecture relies on original x
        
        # FFN Path
        y_ffn = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y_ffn = self.dropout(self.conv2(y_ffn).transpose(-1, 1))
        
        # AFFB Path
        y_affb = self.dropout(self.affb(y))
        
        # 2. Inverse Dimension Scaling for Adaptive Gated Fusion
        # Small V -> strict bias preference; Large V -> smoothed adaptive distribution
        alpha = torch.sigmoid(self.w_gate(y) / (V ** 0.5))
        y_out = alpha * y_ffn + (1 - alpha) * y_affb

        return self.norm2(x + y_out), attn


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

