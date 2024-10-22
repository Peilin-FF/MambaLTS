import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_ssm.Mamba2 import Mamba2

from layers.Embed import DataEmbedding
from layers.Mamba_Family import Mamba2_Layer

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.mamba_layers = nn.ModuleList(
            [
                Mamba2_Layer(
                    Mamba2(
                    d_model = configs.d_model,
                    d_state = configs.d_ff,
                    d_conv = configs.d_conv,
                    expand = configs.expand,
                    headdim = configs.headdim
                ),
                    configs.d_model
                ) 
                for i in range(configs.n_layers)
            ]
        )
        self.n_layers = configs.n_layers

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)

        for i in range(self.n_layers):
            x = self.mamba_layers[i](x)

        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
