import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class CenterNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, dim=3, frozen=False, affine=True, *args, **kwargs):
        super(CenterNorm, self).__init__()
        self.frozen = frozen
        self.num_features = num_features
        self.momentum = momentum
        self.dim = dim
        self.shape = [1 for _ in range(dim)]
        self.shape[1] = self.num_features
        self.affine = affine
        if self.affine:
            self.bias = Parameter(torch.Tensor(*self.shape))
        self.register_buffer('running_mean', torch.zeros(self.num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.zeros_(self.bias)
        self.running_mean.zero_()

    def forward(self, input: torch.Tensor):
        assert input.size(1) == self.num_features and self.dim == input.dim()
        x = input.transpose(0, 1).contiguous().view(self.num_features, -1)
        if self.training and not self.frozen:
            mean = x.mean(-1, keepdim=True)
            xn = x - mean
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
        else:
            xn = x - self.running_mean
        output = xn.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()
        if self.affine:
            output = output + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, frozen={frozen}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    cn = CenterNorm(4,affine=False)
    print(cn)
    print(cn.running_mean.size())
    x = torch.randn(3, 4, 4)*2+1
    zx = x.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(zx.mean(-1))
    y = cn(x)
    zy = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(zy.mean(-1))
    print(cn.running_mean.size())
