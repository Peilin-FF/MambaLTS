import torch
import torch.nn as nn

class PositionalNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(PositionalNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Calculate mean and variance along the feature dimension for each time step
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # (batch, seq_len, 1)
        
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # (batch, seq_len, d_ssm)
        
        # Scale and shift
        y = self.gamma.view(1, 1, -1) * x_hat + self.beta.view(1, 1, -1)
        
        return y