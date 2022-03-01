import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelCombine(nn.Module):
    def __init__(self, num_hiddens, combine_type, num_channels=None):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.dim_query = num_hiddens
        self.combine_type = combine_type
        self.num_channels = num_channels
        self.num_heads = 4

        if combine_type in ['mean']:
            pass
        elif combine_type == 'transform':
            self.transformation = nn.Linear(num_hiddens * num_channels, num_hiddens)
        elif combine_type == 'attn':
            self.attention = nn.Sequential(
                nn.Linear(num_hiddens, num_hiddens), nn.Tanh(), nn.Linear(num_hiddens, 1, bias=False)
            )
        elif combine_type == 'multi_head':
            self.attentions = nn.ModuleList()
            # self.weights = nn.ModuleList()
            for _ in range(self.num_heads):
                self.attentions.append(nn.Sequential(
                    nn.Linear(num_hiddens, num_hiddens), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(num_hiddens, 1, bias=False)
                ))
        elif combine_type == 'global_attn':
            self.transformation = nn.Linear(num_hiddens * num_channels, num_hiddens)
            self.global_attention = nn.Sequential(
                nn.Linear(num_hiddens*2, num_hiddens), nn.Tanh(), nn.Linear(num_hiddens, 1, bias=False)
            )

    
    def forward(self, h):
        # h \in (C, N, d)
        if self.combine_type == 'mean':
            h = h.mean(dim=0)
        elif self.combine_type == 'transform':
            # (C, N, d) to (N, C, d) to (N, C*d)
            h = h.permute(1, 0, 2).flatten(start_dim=1)
            h = self.transformation(h)
        elif self.combine_type == 'attn':
            attn = F.softmax(self.attention(h), dim=0)
            h = (attn * h).sum(dim=0)
        elif self.combine_type == 'multi_head':
            attn = [F.softmax(attention(h), dim=0) for attention in self.attentions]
            attn = torch.stack(attn, dim=0).mean(dim=0)
            h = (attn * h).sum(dim=0)
        elif self.combine_type == 'global_attn':
            global_h = h.permute(1, 0, 2).flatten(start_dim=1)  # (N, C*d)
            global_h = self.transformation(global_h)  # (N, d)
            global_h = global_h.repeat(self.num_channels, 1, 1) # (C, N, d)
            local_global = torch.cat((h, global_h), dim=-1) # (C, N, 2d)
            attn = self.global_attention(local_global) # (C, N, 1)
            h = (h*attn).sum(dim=0)
        return h