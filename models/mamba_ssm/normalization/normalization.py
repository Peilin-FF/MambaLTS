import argparse
import torch.nn as nn
from .center_normalization import CenterNorm
from .scale_normalization import ScaleNorm
from .batch_group_normalization import BatchGroupNorm
#from .iterative_normalization import IterNorm
from .iterative_normalization_FlexGroup import IterNorm
from .dbn import DBN
from .pcaWhitening import PCAWhitening
from .qrWhitening import QRWhitening
from .eigWhitening import EIGWhitening

from .iterative_normalization_FlexGroupSigma import IterNormSigma
from .dbnSigma import DBNSigma
from .pcaWhiteningSigma import PCAWhiteningSigma
from .qrWhiteningSigma import QRWhiteningSigma
from .eigWhiteningSigma import EIGWhiteningSigma

from .dbn import DBN
from .dbnSigma import DBNSigma
from .qrWhitening import QRWhitening
from .qrWhiteningSigma import QRWhiteningSigma

from .CorItN import CorItN
from .CorItN_SE import CorItN_SE, CorItNSigma_SE
from .CorDBN import CorDBN
from .batch_group_whitening import BatchGroupItN, BatchGroupDBN
from .instance_group_whitening import InstanceGroupItN
from .instance_group_whitening_SVD import InstanceGroupSVD
from .iterative_normalization_instance import ItNInstance


from .NormedConv import IdentityModule, WN_Conv2d, CWN_Conv2d, Pearson_Conv2d, CWN_One_Conv2d,WSN_Conv2d, OWN_Conv2d, OWN_CD_Conv2d, \
    ONI_Conv2d, SN_Conv2d, SNConv2d, ONI_ConvTranspose2d, ONI_Linear, CWN_Linear, Pearson_Linear, CWN_One_Linear, WSN_Linear
from .utils import str2dict


def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    if num_groups>num_features:
        print('------arrive maxum groub numbers of:', num_features)
        num_groups=num_features
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)


def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _BatchNorm(num_features, dim=3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    return (nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d)(num_features, eps=eps, momentum=momentum, affine=affine,
                                                            track_running_stats=track_running_stats)


def _InstanceNorm(num_features, dim=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, *args,
                  **kwargs):
    return (nn.InstanceNorm2d if dim == 4 else nn.InstanceNorm1d)(num_features, eps=eps, momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats)

def _Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *args, **kwargs):
    """return first input"""
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()

def _Identity_fn(x, *args, **kwargs):
    """return first input"""
    return x


class _config:
    norm = 'Center'
    norm_cfg = {}
    norm_methods = {'No': _IdentityModule, 'BN': _BatchNorm, 'GN': _GroupNorm, 'LN': _LayerNorm, 'IN': _InstanceNorm, 'CN': CenterNorm,
                    'Center': CenterNorm, 'Scale': ScaleNorm,
                    'None': None, 'BGN': BatchGroupNorm, 'BGWItN': BatchGroupItN, 'BGWDBN': BatchGroupDBN, 'IGWItN': InstanceGroupItN,
                    'IGWSVD': InstanceGroupSVD, 'DBN': DBN, 'PCA': PCAWhitening, 'QR': QRWhitening, 'ItN': IterNorm, 'ItNIns': ItNInstance,
                    'DBNSigma': DBNSigma, 'PCASigma': PCAWhiteningSigma, 'QRSigma': QRWhiteningSigma, 'ItNSigma': IterNormSigma,
                    'EIG': EIGWhitening, 'EIGSigma': EIGWhiteningSigma,
                    'DBN': DBN, 'DBNSigma': DBNSigma,
                    'QR': QRWhitening, 'QRSigma': QRWhiteningSigma,
                    'CorItN': CorItN, 'CorDBN': CorDBN, 'CorItN_SE': CorItN_SE, 'CorItNSigma_SE': CorItNSigma_SE
                    }

    normConv = 'ONI'
    normConv_cfg = {}
    normConv_methods = {'No': _Conv2d, 'WN': WN_Conv2d, 'CWN': CWN_Conv2d, 'WSN': WSN_Conv2d, 'OWN': OWN_Conv2d, 'OWN_CD': OWN_CD_Conv2d,
                        'ONI': ONI_Conv2d, 'SN': SN_Conv2d, 'NSN': SNConv2d, 'CWN_One': CWN_One_Conv2d, 'Pearson': Pearson_Conv2d}

def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Normalization Options')
    group.add_argument('--norm', default='No', help='Use which normalization layers? {' + ', '.join(
        _config.norm_methods.keys()) + '}' + ' (defalut: {})'.format(_config.norm))
    group.add_argument('--norm-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    group.add_argument('--normConv', default='No', help='Use which weight normalization layers? {' + ', '.join(
        _config.normConv_methods.keys()) + '}' + ' (defalut: {})'.format(_config.normConv))
    group.add_argument('--normConv-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    return group

def getNormConfigFlag():
    flag = ''
    flag += _config.norm
    if str.find(_config.norm, 'GW')>-1 or str.find(_config.norm, 'GN') > -1:
        if _config.norm_cfg.get('num_groups') != None:
            flag += '_NG' + str(_config.norm_cfg.get('num_groups'))
    if str.find(_config.norm,'ItN') > -1:
        if _config.norm_cfg.get('T') != None:
            flag += '_T' + str(_config.norm_cfg.get('T'))
        if _config.norm_cfg.get('num_channels') != None:
            flag += '_NC' + str(_config.norm_cfg.get('num_channels'))

    if str.find(_config.norm,'DBN') > -1 or str.find(_config.norm,'QR') > -1 or str.find(_config.norm,'PCA') > -1 or str.find(_config.norm,'EIG') > -1:
        flag += '_NC' + str(_config.norm_cfg.get('num_channels'))
    if _config.norm_cfg.get('affine') == False:
        flag += '_NoA'
    if _config.norm_cfg.get('momentum') != None:
        flag += '_MM' + str(_config.norm_cfg.get('momentum'))
    #print(_config.normConv_cfg)
    #print(flag)
    flag += '_' + _config.normConv
    if _config.normConv == 'ONI' or _config.normConv == 'SN' or _config.normConv == 'NSN':
        if _config.normConv_cfg.get('T') != None:
            flag += '_T' + str(_config.normConv_cfg.get('T'))

    if _config.normConv == 'ONI' or str.find(_config.normConv, 'OWN') > -1:
        if _config.normConv_cfg.get('norm_groups') != None:
            flag += '_G' + str(_config.normConv_cfg.get('norm_groups'))
    if _config.normConv == 'ONI' or str.find(_config.normConv, 'CWN') > -1 or str.find(_config.normConv, 'WSN') > -1 \
           or _config.normConv == 'Pearson' or _config.normConv == 'WN' or str.find(_config.normConv, 'OWN') > -1:
        if _config.normConv_cfg.get('NScale') != None:
            flag += '_NS' + str(_config.normConv_cfg.get('NScale'))
        if _config.normConv_cfg.get('adjustScale') == True:
            flag += '_AS'
    return flag

def setting(cfg: argparse.Namespace):
    print(_config.__dict__)
    for key, value in vars(cfg).items():
        #print(key)
        #print(value)
        if key in _config.__dict__:
            setattr(_config, key, value)
    #print(_config.__dict__)
    flagName =  getNormConfigFlag()
    print(flagName)
    return flagName


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == 'None':
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)

def NormConv(*args, **kwargs):
    kwargs.update(_config.normConv_cfg)
    return _config.normConv_methods[_config.normConv](*args, **kwargs)
