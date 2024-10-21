import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        N, L, C = x.shape  # Assuming input shape is (batch, seqlen, d_ssm)
        G = self.num_groups
        assert C % G == 0, "num_features must be divisible by num_groups"
        group_size = C // G

        x = x.view(N, L, G, group_size)  # Reshape to (batch, seqlen, num_groups, group_size)

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        x_hat = x_hat.view(N, L, C)

        y = self.gamma.view(1, 1, C) * x_hat + self.beta.view(1, 1, C)
        return y
