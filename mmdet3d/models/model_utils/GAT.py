# -*- coding: utf-8 -*-
"""
@Time ： 2021/12/28 12:45

@File ：GAT.py.py
@IDE ：PyCharm


"""
import numpy
from open3d import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
#from open3d import *
from mmdet3d.models.model_utils import Graph_based_module

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x
class GAT_Batch(nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_Batch, self).__init__()
        self.dropout = dropout
        self.attentions =nn.ModuleList( [GraphAttentionLayerBatch(nfeat, nhid, dropout=dropout,alpha=alpha,concat=True) for _ in range(nheads)])
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i),attention)
        self.out_att = GraphAttentionLayerBatch(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions], dim=2) #att(x,adj) ->B*n*outfeature ->concat  B*n*(outfeature*nhead)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x,adj)
        return x


# class GraphAttentionLayerBatch_GraphFPN(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayerBatch_GraphFPN, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).cuda()
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).cuda()
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=False)
#
#     def forward(self, h, adj):
#         Wh = torch.bmm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh).cuda()
#
#         zero_vec = -9e15*torch.ones_like(e).cuda()
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)
class GraphContextualNetwork(nn.Module):
    def __init__(self, senet_channel, danet_channel,nfeat, nhid, nclass, dropout, alpha, nheads, radius ):
        super(GraphContextualNetwork, self).__init__()
        self.SENet = Graph_based_module.SENet(senet_channel, senet_channel)
        self.DANet = Graph_based_module.DANetHead(danet_channel, danet_channel)
        self.GAT = GAT(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout, alpha=alpha, nheads=nheads)
        self.radius = radius
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.concat_linear = torch.nn.Linear(nfeat*2, nfeat)
    def forward(self, in_features, aggregated_points):#8*128*256, 8*256*3
        batch_size = in_features.shape[0]
        adj = get_batch_graph_from_proposals(aggregated_points, self.radius)
        nor_adj = normalization_adj_batch(adj)
        features = in_features.transpose(1,2)#8*256*128
        GAT_NET = []
        for i in range(batch_size):
            nor_adj_, feature = nor_adj[i], features[i]
            GAT_NET_ = self.GAT(feature, nor_adj_).unsqueeze(0)
            GAT_NET.append(GAT_NET_)
        GAT_NET = torch.cat(GAT_NET, 0)
        #net = torch.cat([GAT_NET, features], 2)#8*256*256
        #net = self.concat_linear(net)#8*256*128
        net = self.gamma * GAT_NET_ + features
        net = net.transpose(1,2)#8*128*256
        net = self.SENet(net)
        net = self.DANet(net)
        return net


class GraphAttentionLayerBatch(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayerBatch, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, input, adj):
        h = torch.matmul(input,self.W) #Batch*n*feature self.W在cuda1上但是input在cuda0上
        N = h.size()[1]#N是点的数量
        Batch_size = h.size()[0]
        a_input = torch.cat([h.repeat(1,1,N).view(Batch_size*N*N,-1),h.repeat(1,N,1).view(Batch_size*N*N, -1)],dim=1).view(
            Batch_size,N,-1,2*self.out_features)#B*N*N*(2*feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))# B*N*N*(2*features) matmul 2*features*1  ->B*N*N

        zero_vec = -9e15 * torch.ones_like(e)#B*N*N
        attention = torch.where(adj > 0, e, zero_vec)#B*N*N
        attention = F.softmax(attention, dim=2)#B*N*N
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)#B*N*N matmul B*N*feature
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#     def forward(self, input, adj):
#         h = torch.mm(input, self.W)
#         N = h.size()[0]
#         a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2*self.out_features)
#         #[N, out_features*N]-(view)-> [N*N, out_features]
#         # [[1,2],   N*(feature*N) 111222333 |123123123  N*N*2*feature
#         #  [3,4]] ->   [[1,2,1,2], -> [[1,2],[1,2],[3,4],[3,4]]
#         #              [3,4,3,4]]
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # [N*N*2out_features] mul 2out_features*1
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj>0, e, zero_vec)
#         attention = F.softmax(attention, dim=1 )
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, h)#attention的列代表着线性组合的权重
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).cuda()
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=False)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh).cuda()

        zero_vec = -9e15*torch.ones_like(e).cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
# points = np.random.rand(1000,3)
# point_cloud = PointCloud()
# point_cloud.points = Vector3dVector(points)
# pcd_tree = geometry.KDTreeFlann(point_cloud)
# [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[200], 10)
# print(k)
# print(idx)
def get_points_distance(point1, point2,k):
    point1 = np.tile(point1 ,k).reshape((k,-1))
    return np.linalg.norm((point2-point1),axis=1)
def build_graph_batch_adj(xyz, k=16):
    batchsize = xyz.shape[0]
    num_points = xyz.shape[1]
    adj = np.zeros((batchsize, num_points, num_points))
    points = xyz.cpu().detach().numpy()
    for pointcloud_id in range(batchsize):
        the_points = points[pointcloud_id]
        point_cloud = PointCloud()
        point_cloud.points = Vector3dVector(the_points)
        pcd_tree = geometry.KDTreeFlann(point_cloud)
        idx = list(map(lambda x:np.array(x[1]),(map(pcd_tree.search_knn_vector_3d, point_cloud.points, [k]*num_points))))
        #list()->256个元素 每个元素是一个tuple，第二个值表示idx
        #calculate the index of x and y respectively
        x =np.arange(num_points).repeat(k)
        y =np.array([out for i_out in idx  for out in i_out])
        pointcloud_id  = [pointcloud_id]*x.shape[0]
        adj[pointcloud_id, x , y] = 1
        #adj[pointcloud_id,y,x] = 1
    return adj

def build_graph_adj(xyz,k=16):
    number_points = xyz.shape[0]
    adj = np.zeros((number_points, number_points))
    '''输入点云的位置以及K值或者最近的R距离内作为邻居'''
    points = xyz.numpy()
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(points)
    pcd_tree = geometry.KDTreeFlann(point_cloud)
    for i in range(number_points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], 16)
        #adj[i][idx] =get_points_distance(points[i],points[idx],k)
        adj[i][list(idx)] = 1
        #adj[:,i][list(idx)] = 1
    return adj
def normalization_adj_batch_cpu(A):
    num_points = A.shape[1]
    batchsize = A.shape[0]
    D_hat = np.sum(A,axis=2)
    D_hat = list(map(np.diag, D_hat))
    D_hat = np.concatenate(D_hat,0).reshape(batchsize,num_points,-1)
    return np.matmul(np.linalg.inv(D_hat),A)
def normalization_adj_batch(A):#输入Batch*N*(tensor) -》normalize -》can use cuda
    device = A.device
    num_points = A.shape[1]
    batchsize = A.shape[0]
    D_hat = torch.sum(A, axis=2).to(device)
    D_hat = list( map(torch.diag, D_hat))
    D_hat = torch.cat(D_hat,0).to(device)
    D_hat = D_hat.reshape(batchsize, num_points,-1)
    #D_hat = torch.diag(D_hat)
    return torch.bmm (D_hat.inverse(),A)
def normalization_adj(A):
    num_points = A.shape[0]
    I = np.eye(num_points)
    A_hat = A
    D_hat = np.array(np.sum(A_hat, axis=0))
    D_hat= np.power(D_hat,-1).flatten()
    D_hat_inv = np.diag(D_hat)
    return D_hat_inv.dot(A_hat)
    #return (D_hat**-1).dot( A_hat)会产生inf，差评！
class text_linear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(text_linear, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature)
    def forward(self, x):
        x = self.linear(x)
        return x
from sklearn.neighbors import NearestNeighbors
def get_batch_graph_from_proposals(proposals, radius):#B*N*3, r
    device = proposals.device
    adj_batch = [get_graph_from_proposals(proposal, radius) for proposal in proposals]
    result_adj_batch =  torch.from_numpy(numpy.array(adj_batch) )
    return  result_adj_batch.to(device)
def get_graph_from_proposals(proposals_coor, radius):#256*3
    '''
    input
    proposals_coor: N*3
    radius: int
    output
    adj: N*N

    '''
    if type(proposals_coor) is torch.Tensor:
        if proposals_coor.is_cuda:
            proposals_coor = proposals_coor.cpu()
        proposals_coor = proposals_coor.detach().numpy()
    proposals_xyz = proposals_coor.tolist()
    neigh = NearestNeighbors(n_neighbors=16, radius=radius)
    neigh.fit(proposals_xyz)
    adj = neigh.radius_neighbors_graph(proposals_xyz).toarray()
    return adj

if __name__ == "__main__":
    device_ids = [0,1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    proposals = torch.randn(8, 256, 3).to(device)
    radius = 2
    features = torch.randn(8, 128, 256).to(device)
    GCN = GraphContextualNetwork(128, 128, 128, 128, 128, 0.4, 0.2, 4, radius)
    GCN = nn.DataParallel(GCN).to(device)
    net = GCN(features, proposals)
    # adj = get_batch_graph_from_proposals(proposals, radius)
    #
    #
    # normalize_adj = normalization_adj_batch(adj, adj.device)
    # print(adj)
    # input = torch.ones(size=(16, 16)).to(device)
    # linear = text_linear(16, 32)
    # linear = nn.DataParallel(linear).to(device)
    # out = linear(input)
    # print(out)
    xyz = torch.randn(4,256,3).to(device)
    #adj = build_graph_batch_adj(xyz, k=16)
    adj = get_batch_graph_from_proposals (xyz, radius)
    #adj = torch.from_numpy(adj).float()
    #adj_hat = torch.from_numpy( normalization_adj_batch_cpu(adj)).to(device)
    adj_normalize = normalization_adj_batch(adj)
    feature = torch.randn(size=(4,256,64)).to(device)
    GAN_batch = GAT_Batch(64,128,128,0.4,0.2,4)
    GAN_batch = nn.DataParallel(GAN_batch).to(device)
    batch_layer = GraphAttentionLayerBatch(64,128,dropout=0.4,alpha=0.2,concat=True)
    batch_layer = nn.DataParallel(batch_layer)
    batch_layer.to(device)
    # GAT_Batch = GAT_Batch(nfeat=64,nhid=128, nclass=128,dropout=0.4,alpha=0.2,nheads=4)
    # GAT_Batch = nn.DataParallel(GAT,device_ids=device_ids)
    # GAT_Batch.to(device)
    output = GAN_batch(feature, adj_normalize)
    print(output)
    # the_output = GAT_Batch(feature, adj_hat)
    # print("the_output is : {}".format(the_output.shape))
    # print(output.shape)
    # batch_xyz = torch.randn(16,256,3)
    # adj = build_graph_batch_adj(batch_xyz, k=16)
    # adj = torch.from_numpy(adj).float()
    # adj_hat = normalization_adj_batch(adj)
    # print(adj_hat.shape)
    #
    # #-----------------------------------#
    #
    # GAT = GAT( 128, 128, 128, dropout=0.4, alpha=0.2, nheads=4).cuda()
    # new_net = torch.randn(16,256,128).cuda()
    # xyz = torch.randn(16,256,3)
    # adj = build_graph_batch_adj(xyz, k=16)
    # adj = torch.from_numpy(adj).float().cuda()
    # adj_hat = normalization_adj_batch(adj)
    # #new_net and adj_hat prepared
    # print("get_adj_hat")
    # after_GAN_net = list(map(GAT, new_net, adj_hat))
    # #new_net = GAT(new_net, adj_hat)
    # after_GAN_net = torch.cat(after_GAN_net,0).reshape(16,256,128)
    # print(after_GAN_net)
    # print("end")
    # input = torch.zeros(size=(16,64,64))
    # W = torch.zeros(size=(64,32))
    # out = torch.matmul(input,W)
    # print(out.shape)
