import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Calculate mean and variance
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], unbiased=False, keepdim=True)
        
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        y = self.gamma.view(1, 1, -1) * x_hat + self.beta.view(1, 1, -1)
        
        return y