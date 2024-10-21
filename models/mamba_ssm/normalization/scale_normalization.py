import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ScaleNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, dim=3, eps=1e-5, frozen=False, affine=True, *args, **kwargs):
        super(ScaleNorm, self).__init__()
        self.frozen = frozen
        self.num_features = num_features
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.shape = [1 for _ in range(dim)]
        self.shape[1] = self.num_features
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
        self.register_buffer('running_std', torch.zeros(self.num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
        self.running_std.fill_(1)

    def forward(self, input: torch.Tensor):
        assert input.size(1) == self.num_features and self.dim == input.dim()
        x = input.transpose(0, 1).contiguous().view(self.num_features, -1)
        if self.training and not self.frozen:
            std = x.std(-1, keepdim=True) + self.eps
            xn = x/std
            self.running_std = (1. - self.momentum) * self.running_std + self.momentum * std.data
        else:
            xn = x/self.running_std
        output = xn.view(input.size(1), input.size(0), *input.size()[2:]).transpose(0, 1).contiguous()
        if self.affine:
            output = output * self.weight
        return output

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, frozen={frozen}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    cn = ScaleNorm(4,affine=False)
    print(cn)
    print(cn.running_std.size())
    x = torch.randn(3, 4, 2, 2)
    zx = x.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(zx.std(-1))
    #cn.eval()
    y = cn(x)
    zy = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(zy.std(-1))
    print(cn.running_std.size())
