import os, sys
import logging
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from fastreid.utils.checkpoint import get_unexpected_parameters_message, get_missing_parameters_message

from .MobileFaceNet import MobileFaceNet, MobileFaceNet_air
from .ShuffleNet_v2 import ShuffleNetV2Backbone

# __init__中的from .build import *, 就是导入这里__all__中的内容
__all__ = [
    'add_mobilefacenet_config', 
    'add_shufflenet_config',
    'build_mobilefacenet', 
    'build_mobilefacenet_air',
    'build_shufflenetv2'
    ]


def add_mobilefacenet_config(cfg):
    _C = cfg
    _C.MODEL.BACKBONE.POOL_TYPE = 'GDConv'      # 'GDConv'  'GAP'  'GMP'  'GAMP_add'
    _C.MODEL.BACKBONE.SETTING_STR = 'MobileFaceNet'
    _C.MODEL.BACKBONE.CBAM = False
    _C.MODEL.BACKBONE.L7SCALE = 1

    return 0


@BACKBONE_REGISTRY.register()
def build_mobilefacenet(cfg):
    """
    Create an instance from config.
    Returns:
        MobileFaceNet: a :class:`MobileFaceNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    pooltype      = cfg.MODEL.BACKBONE.POOL_TYPE
    setting_str   = cfg.MODEL.BACKBONE.SETTING_STR
    CBAM          = cfg.MODEL.BACKBONE.CBAM
    l7scale       = cfg.MODEL.BACKBONE.L7SCALE
    # fmt: on
    model = MobileFaceNet(setting_str=setting_str , CBAM=CBAM, pooltype=pooltype, l7scale=l7scale)

    if pretrain:
        logger = logging.getLogger(__name__)
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            logger.info(f"Random initialized backbone. Please check the pretrain_path: {pretrain_path}")
            return model
        
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info( get_missing_parameters_message(incompatible.missing_keys) )
        if incompatible.unexpected_keys:
            logger.info( get_unexpected_parameters_message(incompatible.unexpected_keys) )

    return model


@BACKBONE_REGISTRY.register()
def build_mobilefacenet_air(cfg):
    """
    Create an instance from config.
    Returns:
        MobileFaceNet_air: a :class:`MobileFaceNet_air` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    pooltype      = cfg.MODEL.BACKBONE.POOL_TYPE
    # setting_str   = cfg.MODEL.BACKBONE.SETTING_STR
    CBAM          = cfg.MODEL.BACKBONE.CBAM
    l7scale       = cfg.MODEL.BACKBONE.L7SCALE
    # fmt: on
    model = MobileFaceNet_air(CBAM=CBAM, pooltype=pooltype, l7scale=l7scale)

    if pretrain:
        logger = logging.getLogger(__name__)
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            logger.info(f"Random initialized backbone. Please check the pretrain_path: {pretrain_path}")
            return model
        
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info( get_missing_parameters_message(incompatible.missing_keys) )
        if incompatible.unexpected_keys:
            logger.info( get_unexpected_parameters_message(incompatible.unexpected_keys) )

    return model


def add_shufflenet_config(cfg):
    _C = cfg
    _C.MODEL.BACKBONE.MODEL_SIZE = '1.0x'
    _C.MODEL.BACKBONE.POOL_LAYER = 'pool_s2'


@BACKBONE_REGISTRY.register()
def build_shufflenetv2(cfg):

    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    model_size = cfg.MODEL.BACKBONE.MODEL_SIZE
    feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
    pool_layer = cfg.MODEL.BACKBONE.POOL_LAYER

    return ShuffleNetV2Backbone(model_size=model_size, pretrained=pretrain, pretrain_path=pretrain_path, 
                                feat_dim=feat_dim, pool_layer=pool_layer)

