import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from typing import List
from torch.autograd.function import once_differentiable

__all__ = ['OWN_CD_Conv2d', 'OWN_CD_Linear']

#  norm funcitons--------------------------------

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

class OWN_CD_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_groups=1, norm_channels=0, NScale=1, adjustScale=False, *args, **kwargs):
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

class OWN_CD_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 norm_groups=1, norm_channels=0, NScale=1, adjustScale=False, *args, **kwargs):
        super(OWN_CD_Linear, self).__init__(in_channels, out_channels, bias)
        if out_channels <= in_channels:
            if norm_channels > 0:
                norm_groups = out_channels // norm_channels
            print('OWN_CD_Linear_Row:----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = OWNNorm_CD(norm_groups=norm_groups)
        
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



if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)

    oni= OWNNorm_CD(norm_groups=2, norm_channels=2)
    w_ = torch.randn(8, 8, 2, 2)

   # print(w_)
    w_.requires_grad_()
    y_ = oni(w_)
    z_ = y_.view(w_.size(0), -1)
    #print(z_.sum(dim=1))
    print(z_.matmul(z_.t()))


