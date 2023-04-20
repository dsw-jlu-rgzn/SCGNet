# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_conv_layer
from mmcv.runner import BaseModule
from torch import nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS


@HEADS.register_module()
class BaseConvBboxHead(BaseModule):
    r"""More general bbox head, with shared conv layers and two optional
    separated branches.

    .. code-block:: none

                     /-> cls convs -> cls_score
        shared convs
                     \-> reg convs -> bbox_pred
    """

    def __init__(self,
                 in_channels=0,
                 shared_conv_channels=(),
                 cls_conv_channels=(),
                 num_cls_out_channels=0,
                 reg_conv_channels=(),
                 num_reg_out_channels=0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(BaseConvBboxHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        assert in_channels > 0
        assert num_cls_out_channels > 0
        assert num_reg_out_channels > 0
        self.in_channels = in_channels
        self.shared_conv_channels = shared_conv_channels
        self.cls_conv_channels = cls_conv_channels
        self.num_cls_out_channels = num_cls_out_channels
        self.reg_conv_channels = reg_conv_channels
        self.num_reg_out_channels = num_reg_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels

        # add cls specific branch
        prev_channel = out_channels
        if len(self.cls_conv_channels) > 0:
            self.cls_convs = self._add_conv_branch(prev_channel,
                                                   self.cls_conv_channels)
            prev_channel = self.cls_conv_channels[-1]

        self.conv_cls = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_cls_out_channels,
            kernel_size=1)
        # add reg specific branch
        prev_channel = out_channels
        if len(self.reg_conv_channels) > 0:
            self.reg_convs = self._add_conv_branch(prev_channel,
                                                   self.reg_conv_channels)
            prev_channel = self.reg_conv_channels[-1]

        self.conv_reg = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_reg_out_channels,
            kernel_size=1)

    def _add_conv_branch(self, in_channels, conv_channels):
        """Add shared or separable branch."""
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers

    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(feats)

        # separate branches
        x_cls = x
        x_reg = x

        if len(self.cls_conv_channels) > 0:
            x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)#(128,20)

        if len(self.reg_conv_channels) > 0:
            x_reg = self.reg_convs(x_reg)
        bbox_pred = self.conv_reg(x_reg)#(128,77)

        return cls_score, bbox_pred
@HEADS.register_module()
class SemanticPredictHead(BaseModule):
    r"""More general bbox head, with shared conv layers and two optional
    separated branches.

    .. code-block:: none

                     /-> cls convs -> cls_score
        shared convs
                     \-> reg convs -> bbox_pred
    """
    # pred_layer_cfg = dict(
    #     in_channels=128, shared_conv_channels=(128, 128), bias=True)
    def __init__(self,
                 linear_channels_z=(),
                 in_channels=0,
                 shared_conv_channels=(),
                 cls_conv_channels=(),
                 num_cls_out_channels=0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(SemanticPredictHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        assert in_channels > 0
        assert num_cls_out_channels > 0

        self.in_channels = in_channels
        self.shared_conv_channels = shared_conv_channels
        self.cls_conv_channels = cls_conv_channels
        self.num_cls_out_channels = num_cls_out_channels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels

        # add cls specific branch
        prev_channel = out_channels
        if len(self.cls_conv_channels) > 0:
            self.cls_convs = self._add_conv_branch(prev_channel,
                                                   self.cls_conv_channels)
            prev_channel = self.cls_conv_channels[-1]

        self.conv_cls = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_cls_out_channels,
            kernel_size=1)
        self.linear_z = self._add_linear_branch(linear_channels_z)


    def _add_linear_branch(self, linear_channels_z):
        in_channels, mid_channels, out_channels = linear_channels_z
        linear_layers = nn.Sequential()
        linear_layers.add_module(
           'linear1', torch.nn.Linear(in_channels, mid_channels)
        )
        linear_layers.add_module(
            'relu1', nn.ReLU()
        )
        linear_layers.add_module(
            'linear2', torch.nn.Linear(mid_channels,out_channels)
        )
        return linear_layers

    def _add_conv_branch(self, in_channels, conv_channels):
        """Add shared or separable branch."""
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers

    def forward(self, feats):

        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(feats)

        # separate branches
        x_cls = x
        x_z = x

        if len(self.cls_conv_channels) > 0:
            x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)#(128,20) 8*18*256
        x_z = x_z.transpose(1,2)#8*128*256->8*256*128
        the_z = self.linear_z(x_z)#8*256*128->8*256*96



        return cls_score, the_z

@HEADS.register_module()
class BaseConvSingleHead(BaseModule):
    r"""More general bbox head, with shared conv layers and
    semantic branches.

    .. code-block:: none

                     /-> cls convs -> cls_score
        shared convs

    """

    def __init__(self,
                 in_channels=0,
                 shared_conv_channels=(),
                 cls_conv_channels=(),
                 num_cls_out_channels=0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(BaseConvSingleHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        assert in_channels > 0
        assert num_cls_out_channels > 0

        self.in_channels = in_channels
        self.shared_conv_channels = shared_conv_channels
        self.cls_conv_channels = cls_conv_channels
        self.num_cls_out_channels = num_cls_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels

        # add cls specific branch
        prev_channel = out_channels
        if len(self.cls_conv_channels) > 0:
            self.cls_convs = self._add_conv_branch(prev_channel,
                                                   self.cls_conv_channels)
            prev_channel = self.cls_conv_channels[-1]

        self.conv_cls = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_cls_out_channels,
            kernel_size=1)


    def _add_conv_branch(self, in_channels, conv_channels):
        """Add shared or separable branch."""
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers

    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(feats)

        # separate branches
        x_cls = x


        if len(self.cls_conv_channels) > 0:
            x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)#(128,20)



        return cls_score
