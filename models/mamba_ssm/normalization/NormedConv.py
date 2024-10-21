"""
Orthogonalization by Newton’s Iteration
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

__all__ = ['WN_Conv2d', 'CWN_Conv2d','Pearson_Conv2d','CWN_One_Conv2d','OWN_Conv2d', 'OWN_CD_Conv2d', 'ONI_Conv2d', 'SN_Conv2d', 'ONI_ConvTranspose2d',
           'WSN_Conv2d','WSN_Linear','ONI_Linear', 'CWN_Linear', 'Pearson_Linear', 'CWN_One_Linear']

#  norm funcitons--------------------------------


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input

class WNorm(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        #std = weight_.std(dim=1, keepdim=True) + 1e-5
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_norm = weight_ / norm
        return weight_norm.view(weight.size())

class CWNorm(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        #std = weight_.std(dim=1, keepdim=True) + 1e-5
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_CWN = weight_ / norm
        return weight_CWN.view(weight.size())

class CWNorm_One(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        #std = weight_.std(dim=1, keepdim=True) + 1e-5
        norm = weight_.norm(dim=1, keepdim=True) + 1e-5
        weight_CWN = weight_ / norm
        weight_CWN_One=weight_CWN + 1/weight_.size(1)
        return weight_CWN_One.view(weight.size())

class WSNorm(torch.nn.Module):
    def forward(self, weight):
        weight_ = weight.view(weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        std = weight_.std(dim=1, keepdim=True) + 1e-5
        weight_CWN = weight_ / std
        return weight_CWN.view(weight.size())

class OWNNorm(torch.nn.Module):
    def __init__(self, norm_groups=1, *args, **kwargs):
        super(OWNNorm, self).__init__()
        self.norm_groups = norm_groups

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        wm = torch.randn(S.shape).to(S)
        #Scales = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        #Us = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for i in range(self.norm_groups):
            U, Eig, _ = S[i].svd()
            Scales = Eig.rsqrt().diag()
            wm[i] = U.mm(Scales).mm(U.t())
        W = wm.matmul(Zc)
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['OWN:']
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)

class OWNNorm_CD(torch.nn.Module):
    def __init__(self, norm_groups=1, *args, **kwargs):
        super(OWNNorm_CD, self).__init__()
        self.norm_groups = norm_groups
        self.eps = 1e-4


    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye_group = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps * eye_group
        wm = torch.randn(S.shape).to(S)
        #Scales = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        #Us = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for i in range(self.norm_groups):
            L = torch.potrf(S[i], upper=False)
            wm[i] = torch.inverse(L)
        W = wm.matmul(Zc)
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['OWN:']
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)

class ONINorm(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps * eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


class ONINorm_colum(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm_colum, self).__init__()
        self.T = T
        self.norm_groups = norm_groups

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc.transpose(1, 2), Zc)
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        #print(S.size())
        #S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = Zc.matmul(B[self.T]).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


#  normedConvs--------------------------------


class WN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('WN_Conv:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = WNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class CWN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(CWN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('CWN_Conv:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class CWN_One_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(CWN_One_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('CWN_One_Conv:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm_One()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class Pearson_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(Pearson_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('Pearson_Conv:---NScale:', NScale, '---adjust:', adjustScale)
        self.eps_ln=1e-5
        self.weight_normalization = CWNorm()

        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        size = input_f.size()
        input_view=input_f.view(size[0], -1)
        mean = input_view.mean(-1, keepdim=True)
        std = input_view.std(dim=-1, keepdim=True)
        input_LN = (input_view - mean) /(std+self.eps_ln)
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_LN.view_as(input_f), weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class WSN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 NScale=1.414, adjustScale=False, *args, **kwargs):
        super(WSN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('WSN_Conv:---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = WSNorm()
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class OWN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, *args, **kwargs):
        super(OWN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        if norm_channels > 0:
            norm_groups = out_channels // norm_channels

        print('OWN_Conv:----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = OWNNorm(norm_groups=norm_groups)

        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class OWN_CD_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, *args, **kwargs):
        super(OWN_CD_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        if norm_channels > 0:
            norm_groups = out_channels // norm_channels

        print('OWN_CD_conv:----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = OWNNorm_CD(norm_groups=norm_groups)

        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class ONI_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 T=5, norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, ONIRow_Fix=False, *args, **kwargs):
        super(ONI_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('ONI channels:--OD:',out_channels, '--ID:', in_channels, '--KS',kernel_size)
        if out_channels <= (in_channels*kernel_size*kernel_size):
            if norm_channels > 0:
                norm_groups = out_channels // norm_channels
            print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        else:
            if ONIRow_Fix:
                print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
                self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
            else: 
                print('ONI_Conv_Colum:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
                self.weight_normalization = ONINorm_colum(T=T, norm_groups=norm_groups)
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
           # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class ONI_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 T=4, norm_groups=1, norm_channels=0, NScale=1, adjustScale=False, *args, **kwargs):
        super(ONI_Linear, self).__init__(in_channels, out_channels, bias)
        if out_channels <= in_channels:
            if norm_channels > 0:
                norm_groups = out_channels // norm_channels
            print('ONI_Linear_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        else:
            print('ONI_Linear_Colum:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm_colum(T=T, norm_groups=norm_groups)

        self.scale_ = torch.ones(out_channels, 1, ).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out

class CWN_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                  NScale=1, adjustScale=False, **kwargs):
        super(CWN_Linear, self).__init__(in_channels, out_channels, bias)
        print('CWN_Linear:----NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm()
        self.scale_ = torch.ones(out_channels, 1, ).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out

class CWN_One_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 NScale=1, adjustScale=False, **kwargs):
        super(CWN_One_Linear, self).__init__(in_channels, out_channels, bias)
        print('CWN_One_Linear:----NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm_One()
        self.scale_ = torch.ones(out_channels, 1, ).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out


class Pearson_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 NScale=1, adjustScale=False, **kwargs):
        super(Pearson_Linear, self).__init__(in_channels, out_channels, bias)
        print('Pearson_Linear:----NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = CWNorm()
        self.eps_ln=1e-5
        self.scale_ = torch.ones(out_channels, 1, ).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        size = input_f.size()
        input_view=input_f.view(size[0], -1)
        mean = input_view.mean(-1, keepdim=True)
        std = input_view.std(dim=-1, keepdim=True)
        input_LN = (input_view - mean) /(std+self.eps_ln)
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_LN, weight_q, self.bias)
        return out


class WSN_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 NScale=1, adjustScale=False, **kwargs):
        super(WSN_Linear, self).__init__(in_channels, out_channels, bias)
        print('WSN_Linear:----NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = WSNorm()
        self.scale_ = torch.ones(out_channels, 1, ).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out



### 1.  SN from the version 1, used in the ImageNet experiments

class SpectralNorm(torch.nn.Module):
    def __init__(self, out_channels, T=1, *args, **kwargs):
        super(SpectralNorm, self).__init__()
        self.T = T
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    #define _l2normalization
    def _l2normalize(self, v, eps=1e-12):
        return v / (torch.norm(v) + eps)

    def max_singular_value(self, W, u=None, Ip=1):
        #xp = W.data
        if not Ip >= 1:
            raise ValueError("Power iteration should be a positive integer")
        if u is None:
            u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
        _u = u
        for _ in range(Ip):
            _v = self._l2normalize(torch.matmul(_u, W.data), eps=1e-12)
            _u = self._l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
        sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
        return sigma, _u

    def forward(self, weight: torch.Tensor):
        w_mat = weight.view(weight.size(0), -1)
        sigma, _u = self.max_singular_value(w_mat, self.u, Ip=self.T)
        #U, sigma_, _ = torch.svd(w_mat)
        self.u.copy_(_u)
        return weight / sigma


class SN_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 T=1,*args, **kwargs):
        super(SN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('SN:--T=', T)
        self.weight_normalization = SpectralNorm(out_channels, T=T)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


#### 2.  New SN module from BIGGAN

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(torch.nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12,*args, **kwargs):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

## Finish of the New SN


###3. SN from the GAN training. pytorch-spectral-normalization-gan: used for GAN training

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm_V3(torch.nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm_V3, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)






#Trans Conv


class ONI_ConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True,
                 T=5, norm_groups=1, NScale=1.414, adjustScale=False):
        super(ONI_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        print('ONI_Column:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
        self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.scale = Parameter(self.scale_)
        else:
            # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('scale', self.scale_)

    def forward(self, input_f: torch.Tensor, output_size=None) -> torch.Tensor:
        output_padding = self._output_padding(input_f, output_size, self.stride, self.padding, self.kernel_size)
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.scale
        out = F.conv_transpose2d(input_f, weight_q, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        return out





if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)

    #oni= OWNNorm_CD(norm_groups=2, norm_channels=2)
    oni= ONINorm(norm_groups=2, norm_channels=0)
    #oni= CWNorm_One()
    w_ = torch.randn(4, 8, 2, 2)

   # print(w_)
    w_.requires_grad_()
    y_ = oni(w_)
    z_ = y_.view(w_.size(0), -1)
    #print(z_.sum(dim=1))
    print(z_.matmul(z_.t()))

    oni_pear=Pearson_Conv2d(3,3,(3,3))
    input=torch.randn(4, 3, 4,4)
    out=oni_pear(input)

#     #oni_ = ONINorm_colum(T=5, norm_groups=1)
#     #oni_ = ONINorm(T=3, norm_groups=1)
#     #oni_ = OWNNorm(norm_groups=2)
#     #oni_ = CWNorm()
#     #oni_ = WNorm()
#     oni_v1 = OWNNorm_CD(norm_groups=2)
#     #oni_v1 = SpectralNorm(2)
#    # oni_v1 = SN(1,1,2)
#     w_ = torch.randn(4, 2, 2, 2)
#     print(w_)
#     w_.requires_grad_()
#     y_ = oni_v1(w_)
#     z_ = y_.view(w_.size(0), -1)
#     #print(z_.sum(dim=1))
#     print(z_.matmul(z_.t()))
#
#     y_.sum().backward()
#     print('w grad', w_.grad.size())
#
#     temp=torch.nn.Conv2d(2,2,(2,2))
#     oni_v3 = SpectralNorm_V3(temp)
#     sss=oni_v3(w_)
#     sss.sum().backward()
#
#     oni_v2= SNConv2d(2,2,(2,2))
#     ss=oni_v2(w_)
# #    conv=ONI_Conv2d(4, 2, 1, adjustScale=True)
#  #   b = conv(w_)
#     #print(b)
