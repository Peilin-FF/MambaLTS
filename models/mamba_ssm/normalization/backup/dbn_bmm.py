import torch
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
# from torch.autograd import Variable
from torch.autograd.function import Function


class _decorr_batch_norm(Function):
    @staticmethod
    def forward(
        ctx, input, n_group, nfeat_group, running_mean, running_proj, eps, momentum,
        eps_first, track_running_stats, affine, weight, bias, training
    ):
        input_size = input.size()
        batch_size = input_size[0]
        input = input.float()  # b x (g x c) x ...
        input_ = input.view(batch_size, n_group, nfeat_group, -1).permute(1, 2, 0, 3)  # b x g x c x .., g x c x b x ...
        input = input_.contiguous().view(n_group, nfeat_group, -1)  # g x c x m
        m = input.size()[2]

        batch_mean = torch.mean(input, 2, keepdim=True)  # g x c x 1
        if track_running_stats:
            if training:  # update running_mean
                torch.add(torch.mul(running_mean, momentum), 1 - momentum, batch_mean, out=running_mean)
            else:  # directly use running mean
                batch_mean = running_mean

        centered = torch.add(input, -1, batch_mean)  # g x c x m

        if not training and track_running_stats:
            du_mat = running_proj
            eig, rotation, u_mat = torch.Tensor(), torch.Tensor(), torch.Tensor()
        else:
            batch_proj = torch.bmm(centered, centered.transpose(1, 2)) / m  # g x c x c

            if eps_first:
                batch_proj = torch.add(batch_proj, eps, torch.eye(nfeat_group, device=input.device))

            eig_splits, rotation_splits = [], []
            for proj in batch_proj:
                # SVD and eign-decomposition: real, symmetrical, positive-semidefinite
                # RuntimeError: Lapack Error gesvd : 31 superdiagonals failed to converge
                rotation, eig, _ = torch.svd(proj)  # rotation: c x c, eig: c
                eig = eig.view(-1, 1)  # c x 1
                rotation_splits.append(rotation)
                eig_splits.append(eig)
            rotation = torch.stack(rotation_splits)  # g x c x c
            eig = torch.stack(eig_splits)  # g x c x 1

            if not eps_first:
                eig = torch.add(eig, eps)

            scale_vec = torch.pow(eig, -0.5)
            u_mat = scale_vec * rotation.transpose(1, 2)  # g x c x c
            du_mat = torch.bmm(rotation, u_mat)
            if track_running_stats:
                torch.add(torch.mul(running_proj, momentum), 1 - momentum, du_mat, out=running_proj)
        output = torch.bmm(du_mat, centered)  # g x c x m

        if affine:
            output = output * weight + bias

        ctx.save_for_backward(eig, rotation, centered, u_mat, output, weight)
        ctx.eps = eps
        ctx.affine = affine
        ctx.n_group = n_group
        ctx.nfeat_group = nfeat_group

        output = output.view(input_.size()).permute(2, 0, 1, 3).view(input_size)
        #               g x c x b x ...             b x g x c x ...

        return output  # b x (g x c) x ...

    @staticmethod
    def backward(ctx, grad_output):
        eig, rotation, centered, u_mat, output, weight = ctx.saved_tensors
        # eig:g x c x 1; rotation:g x c x c; centered:g x c x m;
        # u_mat:g x c x c; output:g x c x m; weight:g x c x 1.
        eps = ctx.eps
        affine = ctx.affine
        n_group = ctx.n_group
        nfeat_group = ctx.nfeat_group

        grad_output_size = grad_output.size()
        batch_size = grad_output_size[0]
        grad_output = grad_output.float()  # b x (g x c) x ...
        grad_output_ = grad_output.view(batch_size, n_group, nfeat_group, -1).permute(1, 2, 0, 3)  # b x g x c x .., g x c x b x ...
        grad_output = grad_output_.contiguous().view(n_group, nfeat_group, -1)  # g x c x m

        if affine:
            grad_output_x = grad_output * weight  # g x c x m
        else:
            grad_output_x = grad_output
        m = grad_output.size()[2]

        x_hat = torch.bmm(u_mat, centered)  # g x c x m

        K = eig - eig.transpose(1, 2) + eps
        proj = [torch.eye(nfeat_group, device=grad_output.device) for _ in range(n_group)]
        proj = torch.stack(proj)
        K = torch.div(torch.eq(proj, 0).float(), K)  # g x c x c

        d_hat_x = torch.bmm(grad_output_x.transpose(1, 2), rotation)  # g x m x c

        f = torch.mean(d_hat_x, 1, keepdim=True)  # g x 1 x c

        FC = torch.bmm(x_hat, d_hat_x) / m  # g x c x c, FC^T

        S = eig * FC  # g x c x c
        FC = FC.transpose(1, 2)

        scale = eig.pow(0.5)  # g x c x 1
        S += scale * FC * scale.view(n_group, 1, -1)  # g x c x c
        S = K.transpose(1, 2) * S  # g x c x c
        S = S + S.transpose(1, 2)

        M_splits = []
        for fc in FC:
            m = torch.diag(fc)  # c
            m = torch.diag(m)  # c x c
            M_splits.append(m)
        M = torch.stack(M_splits)  # g x c x c

        grad_input = d_hat_x - f + torch.bmm(x_hat.transpose(1, 2), (S - M))
        #             g x m x c                  g x m x c
        grad_input = torch.bmm(grad_input, u_mat).transpose(1, 2)  # g x c x m

        grad_weight, grad_bias = None, None
        if affine:
            grad_bias = torch.sum(grad_output, dim=2, keepdim=True)
            grad_weight = torch.sum(output, dim=2, keepdim=True)

        grad_input = grad_input.contiguous().view(grad_output_.size()).permute(2, 0, 1, 3).contiguous().view(grad_output_size)
        #                         g x c x b x ...             b x g x c x ...

        return grad_input, None, None, None, None, None, None, None, None, None, grad_weight, grad_bias, None


class _DecorrBatchNorm(Module):
    def __init__(
            self, num_features, nfeat_group, eps=1e-7, momentum=0.1,
            eps_first=True, track_running_stats=True, affine=False
    ):
        super(_DecorrBatchNorm, self).__init__()
        self.num_features = num_features
        self.nfeat_group = nfeat_group
        self.eps = eps
        self.momentum = momentum
        self.eps_first = eps_first
        self.track_running_stats = track_running_stats
        self.affine = affine

        self.n_group = num_features // nfeat_group

        proj = [torch.eye(self.nfeat_group) for _ in range(self.n_group)]
        self.eye_proj = torch.stack(proj)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.n_group, self.nfeat_group, 1))
            self.register_buffer('running_proj', self.eye_proj)
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_proj', None)

        if self.affine:
            self.weight = Parameter(torch.Tensor(self.n_group, self.nfeat_group, 1))
            self.bias = Parameter(torch.Tensor(self.n_group, self.nfeat_group, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_proj = self.eye_proj
            # nn.init.eye_(self.running_proj)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return  _decorr_batch_norm.apply(
            input, self.n_group, self.nfeat_group, self.running_mean, self.running_proj, self.eps, self.momentum,
            self.eps_first, self.track_running_stats, self.affine, self.weight, self.bias, self.training
        )

    def _check_input_dim(self, input):
            raise NotImplementedError


class DecorrBatchNorm1d(_DecorrBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class DecorrBatchNorm2d(_DecorrBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class DecorrBatchNorm3d(_DecorrBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

if __name__ == '__main__':
    ItN = DecorrBatchNorm2d(16, nfeat_group=2,  momentum=0.1, affine=False)
    print(ItN)
    ItN.train()
    x = torch.randn(32, 16, 8, 8)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))