"""
Reference:  Decorrelated Batch Normalization, CVPR 2018

- Paper:
- Code: https://github.com/princeton-vl/DecorrelatedBN
      or  https://github.com/huangleiBuaa/DecorrelatedBN
"""
import torch.nn
from torch.nn import Parameter


__all__ = ['CorItN', 'CorItNSigma']



class CorItN_Single(torch.nn.Module):
    def __init__(self, num_features, T=5, dim=3, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(CorItN_Single, self).__init__()
        # assert dim == 4, 'CorItN is not support 2D'
        self.eps = eps
        self.eps_bn = 1e-5
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.T = T
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_projection', torch.eye(num_features))


    def forward(self, X: torch.Tensor):
        x = X.transpose(0, 1).contiguous().view(self.num_features, -1)
        d, m = x.size()
        if self.training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            xc = x - mean

            # Calculat std for correlation matrix
            std = xc.std(dim=-1) + self.eps_bn
            std_inv=std.reciprocal_().diag()
            xcs = std_inv.mm(xc)
            # calculate correlation matrix
            sigma = torch.addmm(
                input=torch.eye(self.num_features, device=X.device) * self.eps,
                mat1=xcs,
                mat2=xcs.transpose(0, 1),
                beta=1. / m
            )
            P = [None] * (self.T+1)
            P[0] = torch.eye(d).to(x)
            trace_inv = (sigma * P[0]).sum((0,1), keepdim=True).reciprocal_()
            sigma_N = sigma * trace_inv
            for k in range(self.T):
                P[k+1] = torch.addmm(
                    input=P[k] * 1.5,
                    mat1=torch.matrix_power(P[k], 3),
                    mat2=sigma_N,
                    beta=-0.5
                )
            wm = P[self.T].mul_(trace_inv.sqrt())
            xn = wm.mm(xcs)
            #  maintaining running average for BN and DBN
            projection_corDBN = wm.mm(std_inv)
           # print('Projection_Norm:', projection_corDBN.norm())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * projection_corDBN.data
           # print('runningProjection_Norm:', self.running_projection.norm())
        else:
            xc = x - self.running_mean
            wm = self.running_projection
            xn = wm.mm(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        return Xn

class CorItN(torch.nn.Module):
    def __init__(self, num_features, num_channels=16, T=5, dim=3, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(CorItN, self).__init__()
        # assert dim == 4, 'CorItN is not support 2D'
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1 
        self.num_groups = num_groups
        self.T = T
        self.CorItN_Groups = torch.nn.ModuleList(
            [CorItN_Single(num_features = self.num_channels, T=self.T, eps=eps, momentum=momentum) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.CorItN_Groups.append(CorItN_Single(num_features = num_channels_last, T=self.T, eps=eps, momentum=momentum))

        print('CorItN -------m_perGroup:' + str(self.num_channels) + '---nGroup:' + str(self.num_groups) + '---MM:' + str(self.momentum) + '---Affine:' + str(affine))
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
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
            self.CorItN_Groups[i].running_mean.fill_(0)
            #self.CorItN_Groups[i].running_projection = torch.eye(self.CorItN_Groups[i].running_projection.size()[0]).to(self.CorItN_Groups[i].running_projection)
            self.CorItN_Groups[i].running_projection.fill_(0)


    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.CorItN_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


class CorItNSigma_Single(torch.nn.Module):
    def __init__(self, num_features, T=5, dim=3, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(CorItNSigma_Single, self).__init__()
        # assert dim == 4, 'CorItN is not support 2D'
        self.eps = eps
        self.eps_bn = 1e-5
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.T = T
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_projection', torch.eye(num_features))


    def forward(self, X: torch.Tensor):
        x = X.transpose(0, 1).contiguous().view(self.num_features, -1)
        d, m = x.size()
        if self.training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            xc = x - mean

            # Calculat std for correlation matrix
            std = xc.std(dim=-1) + self.eps_bn
            std_inv=std.reciprocal_().diag()
            xcs = std_inv.mm(xc)
            # calculate correlation matrix
            sigma = torch.addmm(self.eps, torch.eye(self.num_features).to(X), 1. / m, xcs, xcs.transpose(0, 1))
            P = [None] * (self.T+1)
            P[0] = torch.eye(d).to(x)
            trace_inv = (sigma * P[0]).sum((0,1), keepdim=True).reciprocal_()
            sigma_N = sigma * trace_inv
            for k in range(self.T):
                P[k+1] = torch.addmm(1.5, P[k], -0.5, torch.matrix_power(P[k],3),sigma_N)
            wm = P[self.T].mul_(trace_inv.sqrt())
            xn = wm.mm(xcs)
            #  maintaining running average for BN and DBN
            projection_corDBN = sigma.mm(std.diag())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * projection_corDBN.data
        else:
            xc = x - self.running_mean
            sigma = self.running_projection
            P = [None] * (self.T+1)
            P[0] = torch.eye(d).to(x)
            trace_inv = (sigma * P[0]).sum((0,1), keepdim=True).reciprocal_()
            sigma_N = sigma * trace_inv
            for k in range(self.T):
                P[k+1] = torch.addmm(1.5, P[k], -0.5, torch.matrix_power(P[k],3),sigma_N)
            wm = P[self.T].mul_(trace_inv.sqrt())
            xn = wm.mm(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        return Xn

class CorItNSigma(torch.nn.Module):
    def __init__(self, num_features, num_channels=16, T=5, dim=3, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(CorItNSigma, self).__init__()
        # assert dim == 4, 'CorItN is not support 2D'
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1
        self.num_groups = num_groups
        self.T = T
        self.CorItNSigma_Groups = torch.nn.ModuleList(
            [CorItNSigma_Single(num_features = self.num_channels, T=self.T, eps=eps, momentum=momentum) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.CorItNSigma_Groups.append(CorItNSigma_Single(num_features = num_channels_last, T=self.T, eps=eps, momentum=momentum))

        print('CorItNSigma -------m_perGroup:' + str(self.num_channels) + '---nGroup:' + str(self.num_groups) + '---MM:' + str(self.momentum) + '---Affine:' + str(affine))
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
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
            self.CorItNSigma_Groups[i].running_mean.fill_(0)
            #self.CorItNSigma_Groups[i].running_projection = torch.eye(self.CorItNSigma_Groups[i].running_projection.size()[0]).to(self.CorItNSigma_Groups[i].running_projection)
            self.CorItNSigma_Groups[i].running_projection.fill_(0)

    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.CorItNSigma_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    ItN = CorItNSigma(4, num_channels=4, T=5, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    x = torch.randn(4, 4, 2, 2)
    #x = torch.randn(32, 8)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.reset_projection()
    y = ItN(x)
    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))
