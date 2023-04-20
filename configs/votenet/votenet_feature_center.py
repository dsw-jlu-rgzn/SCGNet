_base_ = ['./votenet_8x8_scannet-3d-18class.py']

# model settings, add iou loss
model = dict(
    bbox_head=dict(
        iou_loss=dict(
            type='AxisAlignedIoULoss', reduction='sum', loss_weight=10.0 /
            3.0),
only_feature_center_loss=dict(
            type='CenterLoss', reduction='sum', loss_weight=0.2,num_classes=18, feat_dim=128)
    ))
