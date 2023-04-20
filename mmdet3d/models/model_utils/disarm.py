import torch
from kmeans_gpu import KMeans
from mmcv.cnn import ConvModule
from torch import Tensor
from torch import nn as nn
from typing import Tuple, Union
import numpy as np

from mmdet3d.ops import furthest_point_sample, gather_points, furthest_point_sample_with_dist, calc_square_dist
import torch.nn.functional as F

def gather(indices, features, trans=False):
        def T(x):
            return x.transpose(1, 2).contiguous()
        if trans:
            features = gather_points(T(features), indices)
            return T(features)
        else:
            return gather_points(features, indices)

class RelationAnchor(nn.Module):
    r"""Relation Anchor Sampling Module.
    Sample anchors according to different approaches.
    """

    def __init__(self,
                 sample_approach: str = 'random',
                 num_anchors: int = 15,
                 num_candidate_anchors: int = 64,
                 in_channels=128,
                 conv_ratio=(1/2, 1/2, 1),
                 dropout_ratio=(0.25, 0.25, 0.25),
                 dropout_type='dropout',
                 conv_cfg=dict(type="Conv1d"),
                 norm_cfg=dict(type="BN1d"),
                 act_cfg=dict(type="ReLU")
                 ):
        r"""
        Args:
            num_anchors: number of relation anchors.
            num_candidate_anchors: number of candidate relation anchors.
            approach: `random`, `fps`, `kmeans`, `fps+kmeans`, `fps+diff+kmeans`, `pickup`
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_cdanchors = num_candidate_anchors
        assert sample_approach in ['random', 'nearest', 'all', 'D-FPS', 'F-FPS', 'kmeans', 'OS-FFPS']
        if 'kmeans' in sample_approach:
            self.kmeans = KMeans(
                n_clusters=self.num_anchors,
                max_iter=20,
                distance='euclidean',
                #sub_sampling=sub_sampling,
                max_neighbors=5,
                differentiable=True if 'diff' in sample_approach else False,
            )
        if 'OS' in sample_approach:
            self._build_objectness_mlps(
                in_channels, conv_ratio, dropout_ratio, dropout_type, conv_cfg, norm_cfg, act_cfg
            )
        self.approach = sample_approach

    def _build_objectness_mlps(self, in_channels, conv_ratio, dropout_ratio, dropout_type, conv_cfg, norm_cfg, act_cfg):
        dropout_layer = nn.Dropout if dropout_type == 'dropout' else nn.AlphaDropout
        conv_channels = []
        prev_channels = in_channels
        for k in conv_ratio:
            out_channels = int(prev_channels * k)
            conv_channels.append(out_channels)
            prev_channels = out_channels

        prev_channels = in_channels
        obj_mlp_list = list()
        for i, out_ch in enumerate(conv_channels):
            obj_mlp_list.append(
                dropout_layer(dropout_ratio[i])
            )
            obj_mlp_list.append(
                ConvModule(
                    prev_channels,
                    out_ch,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True,
                )
            )
            prev_channels = out_ch
        obj_mlp_list.append(
            dropout_layer(dropout_ratio[-1])
        )
        obj_mlp_list.append(nn.Conv1d(prev_channels, 1, 1))
        self.obj_mlps = nn.Sequential(*obj_mlp_list)

    def _objectness_mlps_forward(self, features):
        return self.obj_mlps(features)

    def _fps_tools(self, locations: Tensor,
                    features: Tensor,
                    num_anchors: int,
                    objectness: Tensor,
                    strategy: str) -> Tuple[Tensor, Tensor]:
        
        if 'D-FPS' in strategy:
            anchor_indices = furthest_point_sample(locations, num_anchors)
        elif 'F-FPS' in strategy:
            anchor_indices = furthest_point_sample(locations, num_anchors)
        elif 'OS-FFPS' in strategy:
            # objectness aware FPS on feature space
            objectness = objectness.transpose(1, 2) 
            features_l2 = features.permute(0, 2, 1).norm(p=2, dim=2, keepdim=True)
            features_dist = locations + objectness * features_l2
            anchor_indices = furthest_point_sample(features_dist, num_anchors)
        else:
            raise NotImplementedError

        def T(x):
            return x.transpose(1, 2).contiguous()
        anchor_points = gather_points(T(locations), anchor_indices)
        anchor_points = T(anchor_points)
        if features is not None:
            anchor_features = gather_points(features, anchor_indices)
            _anchor_points = anchor_points.detach()
            _anchor_features = anchor_features.detach()
            return _anchor_points, _anchor_features, anchor_indices
        else:
            return anchor_points
    
    def _kmeans_tools(self,
                    locations: Tensor,
                    features: Tensor,
                    num_anchors: int,
                    kmeans_layer: KMeans,
                    approach: str) -> Tuple[Tensor, Tensor]:
        """
        Args:
            aggregated_points: BxNx3
            features: BxFxN
            anchor_points: int
            kmeans_layer: KMeans
        """
        # Sample some initial points with FPS
        if 'D-FPS' in approach:
            init_centroids = self._fps_tools(
                locations, features=None, num_anchors=num_anchors, strategy='D-FPS')
        elif 'F-FPS' in approach:
            init_centroids = self._fps_tools(
                locations=None, features=features, num_anchors=num_anchors, strategy='F-FPS')
        else:
            init_centroids = None

        # Forward on kmeans layer
        return kmeans_layer(locations, features, init_centroids)

    def forward(self, locations, features):
        r"""
        Args:
            locations: bz,num,3
            features: bz,fd,num
        Returns:
            points: bz,num_anchor,3
            features: bz,fd,num_anchor
        """
        anchors = dict()
        if self.approach == "random":
            anchor_locations, anchor_features = locations[:, :self.num_anchors],\
                features[:, :, :self.num_anchors]
        elif self.approach == "all":
            anchor_locations, anchor_features = locations[:, 1:, :], features[:, :, 1:]
        elif self.approach == "nearest":
            _dist = calc_square_dist(locations, locations, norm=False)
            _, anchor_indices = torch.topk(_dist, k=self.num_anchors + 1)
            anchor_indices = anchor_indices[:, :, 1:]
            anchor_locations = self._gather(anchor_indices, locations, True)
            anchor_features = self._gather(anchor_indices, features)
        elif self.approach == "D-FPS" or self.approach == "F-FPS":
            anchor_locations, anchor_features, anchor_indices = self._fps_tools(
                locations, features, self.num_anchors, strategy=self.approach)
        elif self.approach == "kmeans":
            anchor_locations, anchor_features = self._kmeans_tools(
                locations, features, self.num_anchors, self.kmeans, self.approach)
        elif self.approach == 'OS-FFPS':
            obj_scores = self._objectness_mlps_forward(features.detach()).squeeze(dim=1)
            anchors['rn_anchor_objectness'] = obj_scores
            obj_scores = torch.sigmoid(obj_scores)

            pickup_indices = torch.topk(obj_scores, k=self.num_cdanchors, dim=-1)[1].int()
            def T(x):
                return x.transpose(1, 2).contiguous()
            pickup_points = gather_points(T(locations), pickup_indices)
            pickup_points = T(pickup_points)
            pickup_features = gather_points(features, pickup_indices)

            pickup_scores = gather_points(T(obj_scores.unsqueeze(2)), pickup_indices)
            anchor_locations, anchor_features, anchor_indices = self._fps_tools(
                pickup_points, pickup_features, self.num_anchors, pickup_scores, 'OS-FFPS')

        else:
            raise NotImplementedError

        anchors['anchor_locations']=anchor_locations
        anchors['anchor_features']=anchor_features
        anchors['anchor_indices']=anchor_indices

        return anchors


class DisplacementWeight(nn.Module):
    r"""Geometry Encode Module
    """

    def __init__(self,
                 in_channels=128,
                 dropout_type='dropout',
                 conv_chs=(32, 32, 1),
                 conv_cfg=dict(type="Conv1d"),
                 norm_cfg=dict(type="BN1d"),
                 act_cfg=dict(type="Tanh"),
                 ):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels % 2 == 0
        
        self.feature_dis_mlps = self._build_displacement_module(
            in_channels, [], [1/2, 1/2], conv_cfg, norm_cfg, act_cfg, dropout_ratio=[0.25, 0.25], dropout_type=dropout_type
        )
        self.spatial_dis_mlps = self._build_displacement_module(    
            in_channels, [3, 8, 16, 32], [], conv_cfg, norm_cfg, act_cfg, dropout_ratio=None, dropout_type=''
        )
        
        # Get the aggregated weights module
        m = list(self.feature_dis_mlps.modules())[-1]
        out_ch = m.out_channels
        conv_chs = list(conv_chs)
        conv_chs[0] += out_ch
        self._build_aggregated_mlp(conv_chs, conv_cfg, norm_cfg, act_cfg)
        self.bn = torch.nn.BatchNorm1d(1)
        self.softmax = torch.nn.Softmax(dim=3)
           
    def _build_aggregated_mlp(self, conv_chs, conv_cfg, norm_cfg, act_cfg):
        dropout_layer = nn.Dropout
        
        mlp_list = []
        prev_channels = conv_chs[0]
        for out_ch in conv_chs[1:-1]:
            mlp_list.append(
                dropout_layer(0.5)
            )
            mlp_list.append(
                ConvModule(
                    prev_channels,
                    out_ch,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True,
                ))
            prev_channels = out_ch

        mlp_list.append(
            dropout_layer(0.1)
        )
        mlp_list.append(nn.Conv1d(prev_channels, conv_chs[-1], 1))
        self.agg_mlp = nn.Sequential(*mlp_list)

    def _aggregated_weights_forward(self, points):
        return self.agg_mlp(points)

    def _build_displacement_module(self, 
                                    in_channels, conv_chs, conv_ratio,conv_cfg, 
                                    norm_cfg, act_cfg, dropout_ratio=None, dropout_type=''):
        dropout_layer = nn.Dropout if dropout_type == 'dropout' else nn.AlphaDropout
        
        if conv_ratio:
            conv_channels = [in_channels]
            prev_channels = in_channels
            for k in conv_ratio:
                conv_channels.append(int(conv_channels[-1] * k))
        if conv_chs:
            conv_channels = conv_chs
            prev_channels = conv_chs[0]

        mlp_list = list()
        for i, out_ch in enumerate(conv_channels[1:-1]):
            if dropout_ratio:
                mlp_list.append(dropout_layer(dropout_ratio[i]))
            mlp_list.append(
                ConvModule(
                    prev_channels,
                    out_ch,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True,
                )
            )
            prev_channels = out_ch
        
        if dropout_ratio:
            mlp_list.append(dropout_layer(dropout_ratio[-1]))

        out_ch = conv_channels[-1]
        mlp_list.append(nn.Conv1d(prev_channels, out_ch, 1))
        return nn.Sequential(*mlp_list)

    def _feature_displacement_forward(self, features):
        return self.feature_dis_mlps(features)

    def _spatial_displacement_forward(self, locations):
        return self.spatial_dis_mlps(locations)

    def _normlization(self, weights, range_list=[0,1]):
        _min = weights.min(dim=-1)[0]
        _max = weights.max(dim=-1)[0]
        k = (range_list[1] - range_list[0] + 1e-6)/(_max - _min + 1e-6)
        weights = range_list[0] + k.unsqueeze(-1) * (weights - _min.unsqueeze(-1))
        return weights

    def _dis_weight_forward(self, locations, features):
        bz, num, num_a, _ = locations.shape
        feature_dis = self._feature_displacement_forward(features.permute(0, 2, 1, 3).reshape(bz * num, -1, num_a)) # [bz * num, -1, num_a]
        spatial_dis = self._spatial_displacement_forward(locations.permute(0, 1, 3, 2).reshape(bz * num, -1, num_a)) 
        return self._aggregated_weights_forward(torch.cat([spatial_dis, feature_dis], dim=1)).reshape(bz, num, -1, num_a).permute(0, 2, 1, 3)


    def forward(self, locations, features):
        f"""
        Args:
            locations: bz, num_proposals, num_anchors, 3
            features: bz, fd, num_proposals, num_anchors.
        Returns:
            weight: bz, 1, num_proposals, num_anchors
        """
        output = self._dis_weight_forward(locations, features)
        output = self.softmax(torch.tanh(output))
        return output, self._normlization(output, [0,1])
        

class RelationFeatureFusion(nn.Module):
    r"""Relation Feature Fusion module
    """

    def __init__(self,
                 in_channels: int = 256,
                 conv_ratio=(1/2, 1/2, 1),
                 dropout_ratio=(0.25, 0.25, 0.25),
                 conv_cfg=dict(type="Conv1d"),
                 norm_cfg=dict(type="BN1d"),
                 act_cfg=dict(type="ReLU"),
                 ):
        super().__init__()

        conv_channels = []
        prev_channels = in_channels * 2
        for k in conv_ratio:
            out_channels = int(prev_channels * k)
            conv_channels.append(out_channels)
            prev_channels = out_channels

        prev_channels = conv_channels[0]
        relation_mlp_list = list()
        for i, out_ch in enumerate(conv_channels):
            relation_mlp_list.append(
                nn.Dropout(dropout_ratio[i])
            ) 
            relation_mlp_list.append(
                ConvModule(
                    prev_channels,
                    out_ch,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True,
                )
            )
            prev_channels = out_ch
        relation_mlp_list.append(nn.Conv1d(prev_channels, prev_channels, 1))
        self.relation_mlps = nn.Sequential(*relation_mlp_list)

    def forward(self, features):
        r"""
        Args:
            features: B, N, C
        Returns:
            relation: B, C, N
        """
        return self.relation_mlps(features)


class DisARM(nn.Module):
    r"""Displacement Aware Relation module.

    Provied effective and informative context information by 
    sampling relation anchors and fusing the displacement weighted features.

    Args:
        sample_approach: Sample strategy for relation anchors.
        num_anchors: Number of relation anchors.
        num_candidate_anchors: Number of candidate relation anchors.
        in_channels: Number of channels of proposals' features.
        conv_ratio (tuple[int]): The ratio of feature channels.
        dropout_ratio (tuple[int]): The ratio of dropout functions.
        dropout_type: Config of dropout.
            Default: dropout
        conv_cfg (dict): Config of convolution.
            Default: dict(type="Conv1d")
        norm_cfg (dict): Config of normalization.
            Default: dict(type="BN1d")
        act_cfg (dict): Config of action layer.
            Default: dict(type="ReLU")
    """

    def __init__(self,
                 sample_approach: str,
                 num_anchors: int = 15,
                 num_candidate_anchors: int = 64,
                 in_channels=128
                 ):
        super().__init__()
        self.num_anchors = num_anchors
        self.in_channels = in_channels

        self.relation_anchor_sampling = RelationAnchor(
            sample_approach,
            num_anchors=num_anchors,
            num_candidate_anchors=num_candidate_anchors,
            in_channels=in_channels
        )
        self.displacement_weight = DisplacementWeight(
            in_channels=in_channels
        )

        self.relation_mlp = RelationFeatureFusion(
            in_channels=in_channels*2
        )

    def forward(self, proposal_locations: torch.Tensor, proposal_features: torch.Tensor):
        """Displacement Aware Relation module forward.
        Args:
            proposal_locations (torch.Tensor): Coordinate of the proposals in shape (B, N, 3).
            proposal_features (torch.Tensor): Features of the proposals in shape (B, C, N).
        Returns:
            torch.Tensor: features with relational context information.
        """
        bz, fd, num = proposal_features.shape
        num_a = self.num_anchors

        # 1. Relation Anchors Sampling
        # anchors: B, C, num_anchor
        _p_l = proposal_locations.clone()
        _p_f = proposal_features.clone()
        anchors = self.relation_anchor_sampling(_p_l, _p_f)
        anchor_locations, anchor_features = anchors['anchor_locations'], anchors['anchor_features']

        # 2. Displacement Weights
        # weight: B, C, N, num_anchor
        proposal_feat_exp = proposal_features.unsqueeze(dim=-1).expand(bz, fd, num, num_a) # B, C, N, 1
        anchor_features_exp = anchor_features.unsqueeze(dim=-2).expand_as(proposal_feat_exp) # B, C, 1, num_anchor

        # compute residual shift
        # B, N, num_anchor, 3
        proposal_locations_exp = proposal_locations.unsqueeze(dim=2)
        anchor_locations_exp = anchor_locations.unsqueeze(dim=1)

        feature_displacement = proposal_feat_exp - anchor_features_exp
        spatial_displacement = proposal_locations_exp - anchor_locations_exp
        _s_d = spatial_displacement.clone()
        _s_f = feature_displacement.clone()
        _, weight = self.displacement_weight(_s_d, _s_f)


        # 3. Displacement based context feature fusion
        # relation feature: B, C, N
        # concate vec: B, C, N, num_anchor
        cat_vec = torch.cat([proposal_feat_exp, anchor_features_exp], dim=1)
        weight_cat_vec = weight * cat_vec
        relation = self.relation_mlp(torch.sum(weight_cat_vec, dim=-1, keepdim=False))

        
        results = dict()
        results['relation'] = relation
        results.update(anchors)
        return results
