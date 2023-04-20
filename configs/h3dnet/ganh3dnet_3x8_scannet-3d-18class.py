# -*- coding: utf-8 -*-
"""
@Time ： 2021/12/10 21:08
@Auth ： ShuweiDong
@File ：ganh3dnet_3x8_scannet-3d-18class.py.py
@IDE ：PyCharm
@Motto：I am LingLing

"""
_base_ = ['./h3dnet_3x8_scannet-3d-18class.py']

DEGREE = [1, 2, 2, 2, 2, 2, 64]
model = dict(
    rpn_head=dict(
        type='GanVoteHead',
        senet_channel=128,
        danet_channel=128,
        is_global_residual_net=True,
        G_config=dict(features=[96, 256, 256, 256, 128, 128, 128, 3],
                      degrees=DEGREE,
                      support=10
                      ),
        D_config=dict(features=[3, 64, 128, 256, 256, 512]),
        pred_layer_cfg_gan=dict(linear_channels_z=(128, 128, 96),
                                in_channels=128, shared_conv_channels=(128, 128), bias=True),
        no_gradient_G=True,
        no_gradient_D=True,
        new_class=[4, 2, 7, 16, 17, 0, 1, 3],
        first_semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)
    )
)
data = dict(
    samples_per_gpu=4,
)
lr = 0.008*3  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
h3dnet_ckp_dir = "/nfs/volume-92-1/shuweidong_i/the_mmdetection3d/mmdetection3d/checkpoints/" \
                  "h3dnet_3x8_scannet-3d-18class_gan_votehead_backbone_pretrain.pth"
init_cfg = dict(type="pretrained", checkpoint=h3dnet_ckp_dir, )

find_unused_parameters = True