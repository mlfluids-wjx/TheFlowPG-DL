"""TPM network: Fourier Feature Embedding, Temporal Window Attention and the stacked LSTM."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    
class TimeRFF(nn.Module):
    """
    Fourier Feature Embedding (FFE) applied to the time index only:
      phi(t) = sqrt(2/F) * [cos(B t + b), sin(B t + b)]
    where B ~ N(0, sigma^2)
    """
    def __init__(self, out_features=32, sigma=10.0, learnable=False, add_phase=True):
        super().__init__()
        self.F = out_features
        self.add_phase = add_phase
        self.scale = math.sqrt(2.0 / out_features)
        # t is expected with shape [..., 1]
        B = torch.randn(1, out_features) * sigma
        self.B = nn.Parameter(B, requires_grad=learnable)
        self.b = nn.Parameter(2*math.pi*torch.rand(1, out_features), requires_grad=learnable) if add_phase else None

    def forward(self, t):
        # t: [..., 1], normalised to [0, 1]
        proj = t * self.B  # [..., F]
        if self.b is not None:
            proj = proj + self.b
            return self.scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [..., 2F]
        else:
            return self.scale * torch.cos(proj)  # [..., F]

class SEChannelAttn1D(nn.Module):
    """SE-style channel attention over time. x: [B,T,D]"""
    def __init__(self, dim, r=8):
        super().__init__()
        hid = max(dim // r, 4)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        z = x.mean(dim=1)  # [B,D]
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))  # [B,D]
        return x * w.unsqueeze(1)  # [B,T,D]


class TemporalAttn(nn.Module):
    """Temporal attention using the last step as query. h: [B,T,H] -> c: [B,1,H]"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h):
        q = self.q(h[:, -1:, :])                 # [B,1,H]
        k = self.k(h)                             # [B,T,H]
        a = torch.softmax((q * k).sum(-1), dim=1) # [B,T]
        c = (h * a.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B,1,H]
        return c


class LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 offset=1, bidirectional=False, rff_switch=False,
                 rff_features=32, rff_sigma=10.0, rff_learnable=False,
                 attn_r=8, channel_attn_switch=True, temp_attn_switch=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.offset = offset
        self.rff_switch = rff_switch
        self.channel_attn_switch = channel_attn_switch
        self.temp_attn_switch = temp_attn_switch

        if self.rff_switch:
            self.time_embed = TimeRFF(out_features=rff_features,
                                      sigma=rff_sigma,
                                      learnable=rff_learnable,
                                      add_phase=True)
            self.rff_dim = 2 * rff_features
        else:
            self.rff_dim = 1
        lstm_in_dim = input_size + self.rff_dim + 2

        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1

        # channel attention, applied before the LSTM input
        self.chan_attn = SEChannelAttn1D(lstm_in_dim, r=attn_r)

        self.lstm = nn.LSTM(lstm_in_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)

        # temporal attention, applied after the LSTM output
        self.temp_attn = TemporalAttn(hidden_size * num_dirs)

        self.linear = nn.Linear(hidden_size * num_dirs, output_size)

    def forward(self, x, t=None, cond=None, h=None, c=None):
        B, T, _ = x.shape

        if t is None:
            tt = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, T, 1)
            t = tt.expand(B, T, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
            
        if self.rff_switch:
            t_feat = self.time_embed(t)                 # [B,T,2F]
        else:
            t_feat = t                                 # [B,T,1]
        x_in = torch.cat([x, t_feat, cond], dim=-1)
        
        if self.channel_attn_switch:
            x_in = self.chan_attn(x_in)

        o, (h, c) = self.lstm(x_in, (h, c)) if h is not None else self.lstm(x_in)

        if self.temp_attn_switch:
            ctx = self.temp_attn(o)
            y_hat = self.linear(ctx)                    # [B,1,output_size]
        else:
            y_hat = self.linear(o[:, -1:, :])  # last step -> 1-step ahead
        return y_hat, (h, c)
        
class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False