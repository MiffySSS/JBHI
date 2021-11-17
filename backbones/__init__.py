# traditional semantic_segmentation
from .vgg import vgg, vgg_names
#from .resnet import resnet, resnet_names, resnetd_names, resnext_names
from .densenet import densenet, densenet_names
from .dpn import dpn, dpn_names
from .inception import inception, inception_names
from .xception import xception, xception_names
# light-weight semantic_segmentation
from .mobilenet import mobilenet, mobilenet_names
from .espnet import espnet, espnet_names
from .aspp import build_aspp
from .resnet import build_resnet


def build_backbone(backbone, head_plan, output_stride, norm_layer, f_lenth=1):
    return resnet.build_resnet(backbone, head_plan, output_stride, norm_layer, f_lenth=f_lenth)

__all__ = [
    'vgg', 'vgg_names',
    'resnet', #'resnet_names', 'resnetd_names', 'resnext_names',
    'densenet', 'densenet_names',
    'dpn', 'dpn_names',
    'inception', 'inception_names',
    'xception', 'xception_names',

    'mobilenet', 'mobilenet_names',
    'espnet', 'espnet_names',
    'configure_backbone'
]


def configure_backbone(arch, head_plane, **kwargs):
    _backbone = None
    _info = None
    if arch in vgg_names:
        _backbone, _info = vgg(arch, head_plane, **kwargs)
    elif arch in resnet: #resnet_names | resnetd_names | resnext_names:
       _backbone, _info = resnet(arch, head_plane, **kwargs)
    elif arch in densenet_names:
        _backbone, _info = densenet(arch, head_plane, **kwargs)
    elif arch in dpn_names:
        _backbone, _info = dpn(arch, head_plane, **kwargs)
    elif arch in inception_names:
        _backbone, _info = inception(arch, head_plane, **kwargs)
    elif arch in xception_names:
        _backbone, _info = xception(arch, head_plane, **kwargs)
    elif arch in mobilenet_names:
        _backbone, _info = mobilenet(arch, head_plane, **kwargs)
    elif arch in espnet_names:
        _backbone, _info = espnet(arch, head_plane, **kwargs)
    else:
        raise NotImplementedError('backbone {} with head plane {} is unknown currently!'.format(arch, head_plane))

    return _backbone, _info

