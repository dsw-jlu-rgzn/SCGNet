import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
@LOSSES.register_module()
class CenterLoss(nn.Module):
    '''Center loss
    Reference:
        Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.

    '''

    def __init__(self, num_classes=10, feat_dim=2, reduction='sum', loss_weight=1.0, avg_factor=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        #self.use_gpu = use_gpu

        # if self.use_gpu:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        # else:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, centers, weight=None  ):#
        """
        Args:
            x: feature matrix with shape (batch_size, proposal_num,  feat_dim).
            labels: ground truth labels with shape (batch_size, proposal_num).
            centers: the center of each class (class_num, feat_dim)
        """
        device = x.device
        batch_size = x.size(0)
        num_proposal = x.shape[1]
        feature_dim = x.shape[-1]
        new_batch_size = batch_size * num_proposal
        x = x.reshape([-1, feature_dim])#(batch*proposal_dim, feat_dim)
        labels = labels.reshape([-1])#(batch*proposal_dim)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(new_batch_size, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, new_batch_size).t()
        distmat.addmm_(x, centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(new_batch_size, self.num_classes)
        mask = labels.eq(classes.expand(new_batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12)
        loss = torch.sum(loss, dim=1) # B*C -> B
        loss = loss.reshape([batch_size, num_proposal])
        loss = weight_reduce_loss(loss, weight=weight, reduction=self.reduction, avg_factor=self.avg_factor)
        # if self.weight is not None:
        #     weight = weight.float()
        # loss = loss * weight
        # loss = loss.sum()
        center_loss = self.loss_weight * loss
        return center_loss