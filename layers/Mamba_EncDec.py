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
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)

        self.low_pass_cut_freq_param = nn.Parameter(torch.tensor(dim / 2.0 - 0.5, dtype=torch.float32))
        self.high_pass_cut_freq_param = nn.Parameter(torch.tensor(dim / 4.0 - 0.5, dtype=torch.float32))

    def adaptive_freq_pass(self, x_fft, flag="high"):
        _, _, c = x_fft.shape
        freq = torch.fft.fftfreq(c, d=1.0 / c).to(x_fft.device)
        if flag == "high":
            freq_mask = torch.abs(freq) >= self.high_pass_cut_freq_param.to(x_fft.device)
        else:
            freq_mask = torch.abs(freq) <= self.low_pass_cut_freq_param.to(x_fft.device)
        return x_fft * freq_mask.view(1, 1, -1)

    def forward(self, x_in):
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # In inverted time-series modeling x is [B, N, D], and AFFB should mix feature dimension D.
        x_fft = torch.fft.fft(x, dim=-1, norm='ortho')

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
        x = torch.fft.ifft(x_weighted, dim=-1, norm='ortho').real

        return x.to(dtype)


class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.attention_r = attention_r
        self.affb = AdaptiveFourierFilterBlock(d_model, adaptive_filter=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        attn = 1

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.affb(y))

        return self.norm2(x + y), attn


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

