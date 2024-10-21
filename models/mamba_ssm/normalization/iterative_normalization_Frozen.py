"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
"""
import torch.nn
from torch.nn import Parameter

# import extension._bcnn as bcnn

__all__ = ['IterNormFrozen']


class IterNormFrozen_Single(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=3, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNormFrozen_Single, self).__init__()
        # assert dim == 4, 'IterNormFrozen is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_features))


    def forward(self, X):
        d = X.size(1)
        x = X.transpose(0,1).contiguous().view(d, -1)
        x_mean = x - self.running_mean
        xn = self.running_wm.matmul(x_mean)
        y = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        return y

class IterNormFrozen(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=3, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNormFrozen, self).__init__()
        # assert dim == 4, 'IterNormFrozen is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1
        self.num_groups = num_groups
        self.iterNorm_Groups = torch.nn.ModuleList(
            [IterNormFrozen_Single(num_features = self.num_channels, eps=eps, momentum=momentum, T=T) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.iterNorm_Groups.append(IterNormFrozen_Single(num_features = num_channels_last, eps=eps, momentum=momentum, T=T))
         
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.register_buffer("weight", torch.Tensor(*shape))
            self.register_buffer("bias", torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)


    def reset_projection(self):
        for i in range(self.num_groups):
            self.iterNorm_Groups[i].running_mean.fill_(0)
            #self.iterNorm_Groups[i].running_wm = torch.eye(self.iterNorm_Groups[i].running_wm.size()[0]).to(self.iterNorm_Groups[i].running_wm)
            self.iterNorm_Groups[i].running_wm.fill_(0)

    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.iterNorm_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    ItN = IterNormFrozen(16, num_channels=4, T=10, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    #x = torch.randn(32, 64, 14, 14)
    x = torch.randn(32, 16)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

