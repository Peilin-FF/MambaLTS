import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from models.mamba_ssm.normalization.center_normalization import CenterNorm
from models.mamba_ssm.normalization.scale_normalization import ScaleNorm
from models.mamba_ssm.normalization.batch_group_normalization import BatchGroupNorm
from models.mamba_ssm.normalization.iterative_normalization_FlexGroup import IterNorm
from models.mamba_ssm.normalization.dbn import DBN
from models.mamba_ssm.normalization.pcaWhitening import PCAWhitening
from models.mamba_ssm.normalization.qrWhitening import QRWhitening
from models.mamba_ssm.normalization.eigWhitening import EIGWhitening

from models.mamba_ssm.normalization.iterative_normalization_FlexGroupSigma import IterNormSigma
from models.mamba_ssm.normalization.dbnSigma import DBNSigma
from models.mamba_ssm.normalization.pcaWhiteningSigma import PCAWhiteningSigma
from models.mamba_ssm.normalization.qrWhiteningSigma import QRWhiteningSigma
from models.mamba_ssm.normalization.eigWhiteningSigma import EIGWhiteningSigma

from models.mamba_ssm.normalization.dbn import DBN
from models.mamba_ssm.normalization.dbnSigma import DBNSigma
from models.mamba_ssm.normalization.qrWhitening import QRWhitening
from models.mamba_ssm.normalization.qrWhiteningSigma import QRWhiteningSigma

from models.mamba_ssm.normalization.CorItN import CorItN
from models.mamba_ssm.normalization.CorItN_SE import CorItN_SE, CorItNSigma_SE
from models.mamba_ssm.normalization.CorDBN import CorDBN
from models.mamba_ssm.normalization.batch_group_whitening import BatchGroupItN, BatchGroupDBN
from models.mamba_ssm.normalization.instance_group_whitening import InstanceGroupItN
from models.mamba_ssm.normalization.instance_group_whitening_SVD import InstanceGroupSVD
from models.mamba_ssm.normalization.iterative_normalization_instance import ItNInstance
from torch.nn import BatchNorm1d
from torch.nn import LayerNorm
from torch.nn import InstanceNorm1d
from torch.nn import GroupNorm
from models.mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
class NormSelection(nn.Module):
    def __init__(self, norm_type, num_features, num_groups=4, num_channels = 0,momentum=0.1, dim=4, eps=1e-5, frozen=False, affine=True, T=5, track_running_stats=True, mode=0, **kwargs):
        """
        初始化归一化层。

        参数：
            norm_type (str): 选择的归一化方法类型。
            num_features (int): 输入通道数，适用于大多数归一化方法。
            num_groups (int): 组数，适用于GroupNorm。
            eps (float): 防止除零的小数，适用于大多数归一化方法。
            momentum (float): 动量，适用于BatchNorm。
            affine (bool): 是否学习缩放和平移参数。
            track_running_stats (bool): 是否跟踪运行时统计数据，适用于BatchNorm。
            **kwargs: 其他可选参数，供自定义归一化方法使用。
        """
        super(NormSelection, self).__init__()
        self.norm_type = norm_type  # Ensure norm_type is properly initialized
        self.frozen = frozen
        self.num_features = num_features
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.shape = [1 for _ in range(dim)]
        self.shape[1] = self.num_features
        self.affine = affine
        self.num_groups = num_groups
        self.T = T
        self.track_running_stats = track_running_stats  # Initialize track_running_stats
        self.kwargs = kwargs  
        self.num_channels = num_channels
        self.mode = mode
        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
        self.register_buffer('running_std', torch.zeros(self.num_features, 1))

        # 定义归一化方法映射
        self.norm_methods = {
            'BN': lambda: BatchNorm1d(num_features=self.num_features, eps=1e-5, momentum=0.1,affine=True),
            'GN': lambda: GroupNorm(num_groups=self.num_groups, num_channels=self.num_features, eps=self.eps, affine=self.affine),
            #'LN': lambda: LayerNorm(normalized_shape=num_features, eps=self.eps, elementwise_affine=self.affine),
            'IN': lambda: InstanceNorm1d(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats),
            'CN': lambda: CenterNorm(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, **self.kwargs),
            'Scale': lambda: ScaleNorm(num_features=self.num_features, eps=self.eps, **self.kwargs),
            'BGN': lambda: BatchGroupNorm(num_features=self.num_features, num_groups=self.num_groups, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode, **self.kwargs),
            'BGWItN': lambda: BatchGroupItN(num_features=self.num_features, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode, **self.kwargs),
            'BGWDBN': lambda: BatchGroupDBN(num_features=self.num_features, num_groups=self.num_groups, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode, **self.kwargs),
            'IGWItN': lambda: InstanceGroupItN(num_features=self.num_features, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode, **self.kwargs),
            'IGWSVD': lambda: InstanceGroupSVD(num_features=self.num_features, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode, **self.kwargs),
            'DBN': lambda: DBN(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, **self.kwargs),
            'PCA': lambda: PCAWhitening(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, **self.kwargs),
            'QR': lambda: QRWhitening(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, **self.kwargs),
            'ItN': lambda: IterNorm(num_features=self.num_features, num_groups=self.num_groups, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, **self.kwargs),
            'ItNIns': lambda: ItNInstance(num_features=self.num_features, num_groups=self.num_groups, T=self.T, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine, mode=self.mode, **self.kwargs),
            'DBNSigma': lambda: DBNSigma(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'PCASigma': lambda: PCAWhiteningSigma(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'QRSigma': lambda: QRWhiteningSigma(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'ItNSigma': lambda: IterNormSigma(num_features=self.num_features, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'EIG': lambda: EIGWhitening(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'EIGSigma': lambda: EIGWhiteningSigma(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'DBN': lambda: DBN(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'DBNSigma': lambda: DBNSigma(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'CorItN': lambda: CorItN(num_features=self.num_features, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'CorDBN': lambda: CorDBN(num_features=self.num_features, num_channels=self.num_channels, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'CorItN_SE': lambda: CorItN_SE(num_features=self.num_features, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'CorItNSigma_SE': lambda: CorItNSigma_SE(num_features=self.num_features, num_channels=self.num_channels, T=self.T, dim=self.dim, eps=self.eps, momentum=self.momentum, affine=self.affine),
            'None': lambda: None,
        }
        # 初始化归一化层
        self.norm = self._get_norm_layer()

    def _get_norm_layer(self):
        """
        根据 norm_type 获取对应的归一化层实例。
        """
        if self.norm_type not in self.norm_methods:
            raise ValueError(f"Unsupported normalization type '{self.norm_type}'. Available types are: {list(self.norm_methods.keys())}")
        
        norm_layer = self.norm_methods[self.norm_type]()
        return norm_layer

    def forward(self, x):
        """
        前向传播方法。

        参数：
            x (Tensor): 输入张量。

        返回：
            Tensor: 经过归一化层处理后的张量。
        """
        if self.norm is not None:
            return self.norm(x)
        else:
            return x  # 如果没有归一化层，直接返回输入

    def __repr__(self):
        return f"{self.__class__.__name__}(norm_type={self.norm_type}, num_features={self.num_features})"
        

# # 示例输入张量
# import torch

# # Sample input tensor
# x = torch.randn(32, 64, 96)  # (batch_size, num_features, sequence_length)

# # Define and apply each normalization method with specific parameters

# # BatchNorm1d
# bn_norm = NormSelection(norm_type='BN', num_features=64)
# output_bn = bn_norm(x)
# print(f"Method: BN\n", bn_norm.norm)
# print(output_bn.shape)

# # GroupNorm
# gn_norm = NormSelection(norm_type='GN', num_features=64, num_groups=8)
# output_gn = gn_norm(x)
# print(f"Method: GN\n", gn_norm.norm)
# print(output_gn.shape)

# # LayerNorm
# ln_norm = NormSelection(norm_type='LN',num_features=64, eps=1e-5,elementwise_affine=True)
# x=x.permute(0,2,1)
# output_ln = ln_norm(x)
# x=x.permute(0,2,1)
# print(f"Method: LN\n", ln_norm.norm)
# print(output_ln.shape)

# # InstanceNorm1d
# in_norm = NormSelection(norm_type='IN', num_features=64, momentum=0.1)
# output_in = in_norm(x)
# print(f"Method: IN\n", in_norm.norm)
# print(output_in.shape)

# # CenterNorm
# cn_norm = NormSelection(norm_type='CN', num_features=64, momentum=0.2)
# output_cn = cn_norm(x)
# print(f"Method: CN\n", cn_norm.norm)
# print(output_cn.shape)

# # ScaleNorm
# scale_norm = NormSelection(norm_type='Scale', num_features=64)
# output_scale = scale_norm(x)
# print(f"Method: Scale\n", scale_norm.norm)
# print(output_scale.shape)

# # BatchGroupNorm
# bgn_norm = NormSelection(norm_type='BGN',num_features=64, num_groups=8, num_channels=0, dim=3, eps=1e-5, momentum=0.1, affine=True)
# output_bgn = bgn_norm(x)
# print(f"Method: BGN\n", bgn_norm.norm)

# print(output_bgn.shape)

# # BatchGroupItN
# bgwitn_norm = NormSelection(norm_type='BGWItN', num_features=64, num_groups=8, num_channels=0, dim=3, eps=1e-5, momentum=0.1, T=10,affine=True)
# output_bgwitn = bgwitn_norm(x)
# print(f"Method: BGWItN\n", bgwitn_norm.norm)
# print(output_bgwitn.shape)

# # BatchGroupDBN
# bgwdbn_norm = NormSelection(norm_type='BGWDBN', num_features=64, num_groups=8, num_channels=64, dim=3)
# output_bgwdbn = bgwdbn_norm(x)
# print(f"Method: BGWDBN\n", bgwdbn_norm.norm)
# print(output_bgwdbn.shape)

# # InstanceGroupItN
# igwitn_norm = NormSelection(norm_type='IGWItN', num_features=64, num_groups=8, T=10, num_channels=64, dim=3)
# output_igwitn = igwitn_norm(x)
# print(f"Method: IGWItN\n", igwitn_norm.norm)
# print(output_igwitn.shape)

# # InstanceGroupSVD
# igwsvd_norm = NormSelection(norm_type='IGWSVD', num_features=64, num_groups=8, T=10, num_channels=64, dim=3)
# output_igwsvd = igwsvd_norm(x)
# print(f"Method: IGWSVD\n", igwsvd_norm.norm)
# print(output_igwsvd.shape)

# # DBN
# dbn_norm = NormSelection(norm_type='DBN', num_features=64, num_channels=64, dim=3)
# output_dbn = dbn_norm(x)
# print(f"Method: DBN\n", dbn_norm.norm)
# print(output_dbn.shape)

# # PCA
# pca_norm = NormSelection(norm_type='PCA', num_features=64, num_channels=64, dim=3)
# output_pca = pca_norm(x)
# print(f"Method: PCA\n", pca_norm.norm)
# print(output_pca.shape)

# # QR
# qr_norm = NormSelection(norm_type='QR', num_features=64, num_channels=64, dim=3)
# output_qr = qr_norm(x)
# print(f"Method: QR\n", qr_norm.norm)
# print(output_qr.shape)

# # IterNorm
# itn_norm = NormSelection(norm_type='ItN', num_features=64, num_groups=8, num_channels=64, T=10, dim=3)
# output_itn = itn_norm(x)
# print(f"Method: ItN\n", itn_norm.norm)
# print(output_itn.shape)

# # ItNInstance
# itnins_norm = NormSelection(norm_type='ItNIns', num_features=64, num_groups=8, T=10, num_channels=64, dim=3)
# output_itnins = itnins_norm(x)
# print(f"Method: ItNIns\n", itnins_norm.norm)
# print(output_itnins.shape)

# # DBNSigma
# dbnsigma_norm = NormSelection(norm_type='DBNSigma', num_features=64, num_channels=64, dim=3)
# output_dbnsigma = dbnsigma_norm(x)
# print(f"Method: DBNSigma\n", dbnsigma_norm.norm)
# print(output_dbnsigma.shape)

# # PCASigma
# pcasigma_norm = NormSelection(norm_type='PCASigma', num_features=64, num_channels=64, dim=3)
# output_pcasigma = pcasigma_norm(x)
# print(f"Method: PCASigma\n", pcasigma_norm.norm)
# print(output_pcasigma.shape)

# # QRSigma
# qrsigma_norm = NormSelection(norm_type='QRSigma', num_features=64, num_channels=64, dim=3)
# output_qrsigma = qrsigma_norm(x)
# print(f"Method: QRSigma\n", qrsigma_norm.norm)
# print(output_qrsigma.shape)

# # ItNSigma
# itnsigma_norm = NormSelection(norm_type='ItNSigma', num_features=64, num_channels=64, T=10, dim=3)
# output_itnsigma = itnsigma_norm(x)
# print(f"Method: ItNSigma\n", itnsigma_norm.norm)
# print(output_itnsigma.shape)

# # EIG
# eig_norm = NormSelection(norm_type='EIG', num_features=64, num_channels=64, dim=3)
# output_eig = eig_norm(x)
# print(f"Method: EIG\n", eig_norm.norm)
# print(output_eig.shape)

# # EIGSigma
# eigsigma_norm = NormSelection(norm_type='EIGSigma', num_features=64, num_channels=64, dim=3)
# output_eigsigma = eigsigma_norm(x)
# print(f"Method: EIGSigma\n", eigsigma_norm.norm)
# print(output_eigsigma.shape)

# # DBN
# dbn_norm = NormSelection(norm_type='DBN', num_features=64, num_channels=64, dim=3)
# output_dbn = dbn_norm(x)
# print(f"Method: DBN\n", dbn_norm.norm)
# print(output_dbn.shape)

# # DBNSigma
# dbnsigma_norm = NormSelection(norm_type='DBNSigma', num_features=64, num_channels=64, dim=3)
# output_dbnsigma = dbnsigma_norm(x)
# print(f"Method: DBNSigma\n", dbnsigma_norm.norm)
# print(output_dbnsigma.shape)

# # QR
# qr_norm = NormSelection(norm_type='QR', num_features=64, num_channels=64, dim=3)
# output_qr = qr_norm(x)
# print(f"Method: QR\n", qr_norm.norm)
# print(output_qr.shape)

# # QRSigma
# qrsigma_norm = NormSelection(norm_type='QRSigma', num_features=64, num_channels=64, dim=3)
# output_qrsigma = qrsigma_norm(x)
# print(f"Method: QRSigma\n", qrsigma_norm.norm)
# print(output_qrsigma.shape)

# # CorItN
# coritn_norm = NormSelection(norm_type='CorItN', num_features=64, num_channels=64, T=10, dim=3)
# output_coritn = coritn_norm(x)
# print(f"Method: CorItN\n", coritn_norm.norm)
# print(output_coritn.shape)

# # CorDBN
# cordbn_norm = NormSelection(norm_type='CorDBN', num_features=64, num_channels=64, dim=3)
# output_cordbn = cordbn_norm(x)
# print(f"Method: CorDBN\n", cordbn_norm.norm)
# print(output_cordbn.shape)

# # CorItN_SE
# coritn_se_norm = NormSelection(norm_type='CorItN_SE', num_features=64, num_channels=64, T=10, dim=3)
# output_coritn_se = coritn_se_norm(x)
# print(f"Method: CorItN_SE\n", coritn_se_norm.norm)
# print(output_coritn_se.shape)

# # CorItNSigma_SE
# coritn_sigma_se_norm = NormSelection(norm_type='CorItNSigma_SE', num_features=64, num_channels=64, T=10, dim=3)
# output_coritn_sigma_se = coritn_sigma_se_norm(x)
# print(f"Method: CorItNSigma_SE\n", coritn_sigma_se_norm.norm)
# print(output_coritn_sigma_se.shape)

# # None
# none_norm = NormSelection(norm_type='None', num_features=64)
# output_none = none_norm(x)
# print(f"Method: None\n", none_norm.norm)
# print(output_none.shape)