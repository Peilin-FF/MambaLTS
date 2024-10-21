import torch
import torch.nn as nn
from torch.nn import Parameter


class BatchGroupDBN(nn.Module):
    def __init__(self, num_features, num_groups=32, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(BatchGroupDBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups>num_features:
            self.num_groups=num_features
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(self.num_groups, 1))
        self.register_buffer('running_projection', torch.eye(self.num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.reshape(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            x_mean = x - mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            u, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm.data
            y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


class BatchGroupDBNSigma(BatchGroupDBN):
    """
    when evaluation phase, sigma using running average.
    """

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        mean = x.mean(1, keepdim=True) if training else self.running_mean
        x_mean = x - mean
        if training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma.data
        else:
            sigma = self.running_projection
        u, eig, _ = sigma.svd()
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        y = wm.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output




class BatchGroupItN(nn.Module):
    def __init__(self, num_features, num_groups=32, T=5, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(BatchGroupItN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        self.T = T
        if self.num_groups>num_features:
            self.num_groups=num_features
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

       # print('BatchGroupItN --- num_groups=', self.num_groups, '--T=', self.T)
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(self.num_groups, 1))
        self.register_buffer('running_projection', torch.eye(self.num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.reshape(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:self.dim])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        d, m = x.size()
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            x_mean = x - mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            P = [None] * (self.T+1)
            P[0] = torch.eye(d).to(x)
            trace_inv=(sigma * P[0]).sum((0,1), keepdim=True).reciprocal_()
            sigma_N=sigma * trace_inv
            for k in range(self.T):
                P[k+1] = torch.addmm(input=P[k], mat1=torch.matrix_power(P[k], 3), mat2=sigma_N, beta=1.5, alpha=-0.5)
            wm = P[self.T].mul_(trace_inv.sqrt())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm.data
            y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


    def reset_projection(self):
        self.running_mean.fill_(0)
        #self.running_projection = torch.eye(self.running_wm.size()[0]).to(running_wm)
        self.running_projection.fill_(0)

class BatchGroupItNSigma(BatchGroupItN):

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        d, m = x.size()
        mean = x.mean(-1, keepdim=True) if training else self.running_mean
        x_mean = x - mean
        if training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma.data
        else:
            sigma = self.running_projection
        P = [None] * (self.T+1)
        P[0] = torch.eye(d).to(x)
        trace_inv=(sigma * P[0]).sum((0,1), keepdim=True).reciprocal_()
        sigma_N=sigma * trace_inv
        for k in range(self.T):
            P[k+1] = torch.addmm(1.5, P[k], -0.5, torch.matrix_power(P[k],3),sigma_N)
        wm = P[self.T].mul_(trace_inv.sqrt())
        y = wm.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output


if __name__ == '__main__':
    #dbn = BatchGroupDBNSigma(16, 4, affine=False, momentum=1.)
    dbn = BatchGroupItNSigma(16, 4, num_channels=8, T=10, affine=False, momentum=1.)
    x = torch.randn(2, 16,96)*2+1
    print(dbn)
    y = dbn(x)
    print('y size:', y.size())
    y = y.view(y.size(0) * y.size(1) // dbn.num_groups, dbn.num_groups, *y.size()[2:])
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    print('y reshaped:', y.size())
    z = y.matmul(y.t())/y.size(1)
    #print('train mode:', z.diag())
    print('train mode:', z)
    #dbn.reset_projection()
    y = dbn(x)
    dbn.eval()
    y = dbn(x)
    y = y.view(y.size(0) * y.size(1) // dbn.num_groups, dbn.num_groups, *y.size()[2:])
    y = y.transpose(0, 1).contiguous().view(dbn.num_groups, -1)
    z = y.matmul(y.t())/y.size(1)
    #print('eval mode:', z.diag())
    print('eval mode:', z)
    print(__file__)
