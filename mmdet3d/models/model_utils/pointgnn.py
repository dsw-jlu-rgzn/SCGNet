import torch
from torch import nn
from torch_scatter import scatter_max
from torch_cluster import knn_graph
from torch_cluster import radius_graph

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph
import numpy

def get_batch_graph_from_proposals(proposals, radius):#B*N*3, r
    device = proposals.device
    adj_batch = [get_graph_from_proposals(proposal, radius) for proposal in proposals]
    result_adj_batch =  torch.from_numpy(numpy.array(adj_batch) )
    return  result_adj_batch.to(device)

def trasfer_batch_adj_to_edges(adjs):
    if len(adjs.shape) < 3:
        adjs = adjs.unsqueeze(0)
    batch_size = adjs.shape[0]
    edges = []
    for i in range(batch_size):
        adj = adjs[i]
        edge = torch.nonzero(adj)
        flag = torch.clone(edge[:,0])
        edge[:,0] = edge[:,1]
        edge[:,1] = flag
        edges.append(edge)
    return edge

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
    # neigh = NearestNeighbors(n_neighbors=16, radius=radius)
    # neigh.fit(proposals_xyz)
    # adj = neigh.radius_neighbors_graph(proposals_xyz).toarray()
    adj = radius_neighbors_graph(proposals_xyz, radius=radius, include_self=False).toarray()
    return adj

def edge_from_points(points, radius=1, k=8):
    '''
    build a graph from points and return its edges
    :param points: (B, N, C)
    :param radius: float64
    :param k: int64
    :return: (K, 2)
    '''
    if radius > 0:
        edge = radius_graph(x=points, r=radius, loop=False)
    else :
        edge = knn_graph(x=points, k=k, loop=False)
    edge = edge.permute(1,0)
    return edge
def batch_edge_from_points(points, radius=0, k=8):
    '''
    batch version of edge_from_points
    :param points: (B, N, C)
    :param radius: float64
    :param k: int64
    :return: list (B)
    '''
    batch_edge = []
    for the_points in points:
        edge = edge_from_points(points=the_points, radius=radius, k=k)
        batch_edge.append(edge)
    #batch_edge = torch.cat(batch_edge, dim=0)
    return batch_edge
def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
            nn.Linear(Ks[i - 1], Ks[i]),
            nn.ReLU(),
            nn.BatchNorm1d(Ks[i])]
    return nn.Sequential(*linears)


def multi_layer_fc_fn(Ks=[300, 64, 32, 64], num_classes=4, is_logits=False, num_layers=4):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
            nn.Linear(Ks[i - 1], Ks[i]),
            nn.ReLU(),
            nn.BatchNorm1d(Ks[i])
        ]

    if is_logits:
        linears += [
            nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
            nn.Linear(Ks[-1], num_classes),
            nn.ReLU(),
            nn.BatchNorm1d(num_classes)
        ]
    return nn.Sequential(*linears)


def max_aggregation_fn(features, index, l):
    """
    Arg: features: N x dim
    index: N x 1, e.g.  [0,0,0,1,1,...l,l]
    l: lenght of keypoints

    """
    index = index.unsqueeze(-1).expand(-1, features.shape[-1])  # N x 64
    set_features = torch.zeros((l, features.shape[-1]), device=features.device).permute(1, 0).contiguous()  # len x 64
    set_features, argmax = scatter_max(features.permute(1, 0), index.permute(1, 0), out=set_features)
    set_features = set_features.permute(1, 0)
    return set_features


def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
    """
    github.com/tensorflow/models/blob/master/\
        research/object_detection/core/losses.py
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size]. N x 1
      logits: A float32 tensor of shape [batch_size]. N x C
      alpha: A scalar for focal loss alpha hyper-parameter.
      If positive samples number > negtive samples number,
      alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `labels`
    """

    prob = logits.sigmoid()
    labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=prob.shape[1])

    cross_ent = torch.clamp(logits, min=0) - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))
    prob_t = (labels * prob) + (1 - labels) * (1 - prob)
    modulating = torch.pow(1 - prob_t, gamma)
    alpha_weight = (labels * alpha) + (1 - labels) * (1 - alpha)

    focal_cross_entropy = modulating * alpha_weight * cross_ent
    return focal_cross_entropy





class GraphNetAutoCenter(nn.Module):
    def __init__(self, auto_offset=True, auto_offset_MLP_depth_list=[128, 64, 3], edge_MLP_depth_list=[128+3, 128, 128],
                 update_MLP_depth_list=[128, 128, 128]):
        super(GraphNetAutoCenter, self).__init__()
        self.auto_offset = auto_offset
        self.auto_offset_fn = multi_layer_neural_network_fn(auto_offset_MLP_depth_list)
        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn(update_MLP_depth_list)

    def forward(self, input_vertex_features,
                input_vertex_coordinates,
                keypoint_indices,
                edges):
        """apply one layer graph network on a graph. .
        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
        returns: a [N, M] tensor. Updated vertex features.
        """
        # print(f"input_vertex_features: {input_vertex_features.shape}")
        # print(f"input_vertex_coordinates: {input_vertex_coordinates.shape}")
        # print(NOT_USED)
        # print(f"edges: {edges.shape}")

        # Gather the source vertex of the edges
        s_vertex_features = input_vertex_features[edges[:, 0]]
        s_vertex_coordinates = input_vertex_coordinates[edges[:, 0]]

        if self.auto_offset:
            offset = self.auto_offset_fn(input_vertex_features)
            input_vertex_coordinates = input_vertex_coordinates + offset#x_j+deta_x_i

        # Gather the destination vertex of the edges
        d_vertex_coordinates = input_vertex_coordinates[edges[:, 1]]

        # Prepare initial edge features
        edge_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)

        # Extract edge features
        edge_features = self.edge_feature_fn(edge_features)

        # Aggregate edge features
        aggregated_edge_features = max_aggregation_fn(edge_features, edges[:, 1], len(keypoint_indices))

        # Update vertex features
        update_features = self.update_fn(aggregated_edge_features)
        output_vertex_features = update_features + input_vertex_features
        return output_vertex_features

if __name__ == "__main__":
    x = torch.tensor([[1,1],[1,0],[0,0],[-1,-1]])
    edge = knn_graph(x,k=1,loop=False,flow='target_to_source')
    # B, N, C = 4, 50, 3
    # x = torch.randn(( B, N, C)).cuda()
    # #batch = torch.tensor([0, 1 , 2, 3])
    # edge = batch_edge_from_points(x,1)
    # # edge_index = radius_graph(x=x,r=0.3,  loop=False)
    # print(edge)
    # graph_nets = nn.ModuleList()
    # for i in range(3):
    #     graph_nets.append(GraphNetAutoCenter())
    # graph_nets.cuda()
    graph = GraphNetAutoCenter().cuda()
    N, M = 200, 128
    D =  3
    input_vertex_features = torch.randn((N, M)).cuda()
    input_vertex_coordinates = torch.randn((N, D)).cuda()
    keypoint_indices = torch.arange(N).cuda()
    edge = edge_from_points(input_vertex_coordinates)
    batch_input = input_vertex_coordinates.unsqueeze(0)
    adj = get_batch_graph_from_proposals(batch_input,radius=1)
    edges = trasfer_batch_adj_to_edges(adj)
    output = graph(input_vertex_features, input_vertex_coordinates, keypoint_indices, edge)