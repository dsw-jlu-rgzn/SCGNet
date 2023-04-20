import torch.nn as nn
from torch.optim import Adam
import torch
import numpy
from mmcv.cnn import ConvModule
from torch.nn import functional as F

def compute_l2_distance(X):#X:batch*Num*F ->8*256*3 out 8*256*256
    #X = X.transpose(2,1)
    n, m = X.shape[1], X.shape[2]
    H_batch = []
    for i in range(X.shape[0]):
        X_ = X[i]
        G = X_ @ X_.t()
        H = torch.diag(G).repeat([n,1])
        H = -torch.sqrt(H + H.t() - 2*G )
        H_batch.append(H)
    H_batch = torch.stack(H_batch,dim=0)
    return H_batch
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



class GAT(torch.nn.Module):
    """
    The most interesting and hardest implementation is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.

    So I'll focus on imp #3 in this notebook.

    """

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """

    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0  # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1  # attention head dim

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)#每个feature都有一个分数为256*8
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

from mmdet3d.models.dense_heads.base_conv_bbox_head import  BaseConvSingleHead
class GraphRelationModule(nn.Module):
    def __init__(self, in_features=128, neighbors=16, num_proposals=256, num_class=18, feature_center_loss=None):
        self.in_features = in_features
        self.neighbors=neighbors
        self.num_proposals = num_proposals
        self.num_class = num_class
        self.feature_center_loss = feature_center_loss
        super().__init__()
        if feature_center_loss is True:
            self.feature_center = nn.Parameter(torch.randn(self.num_class, self.in_features))
        self.conv_pred_semantic_first = BaseConvSingleHead(
            in_channels=self.in_features,
            shared_conv_channels=(self.in_features, self.in_features),
            bias=True,
            num_cls_out_channels=self.num_class)
        self.relation_fc = nn.Sequential(
            nn.Conv1d(self.in_features, self.in_features, 1),
            nn.ReLU(),
            nn.Conv1d(self.in_features, self.in_features, 1)
        )
        self.sg_conv = nn.Sequential(
            nn.Conv1d(self.in_features, self.in_features*2, 1),
            nn.ReLU(),
            nn.Conv1d(self.in_features*2, self.in_features, 1),
            nn.ReLU()
        )
        self.GAT = GAT(num_of_layers=2,
                       num_heads_per_layer=[4, 1],
                       num_features_per_layer=[self.in_features, self.in_features, self.in_features],
                       add_skip_connection=False,
                       bias=True,
                       dropout=0.2,
                       log_attention_weights=True
                       )
        self.relation_mlp = RelationFeatureFusion(in_channels=self.in_features*2)


    def forward(self, features, xyz=None):#8*128*256, 8*256*3
        cls_scores = self.conv_pred_semantic_first(features)
        cls_scores = cls_scores.transpose(2, 1)  # 8*256*18
        cls_prob = F.softmax(cls_scores, dim=2)
        # z = self.relation_fc(features).transpose(2, 1)  # 8*256*128
        # eps = torch.bmm(z, z.transpose(2, 1))
        # _, indices = torch.topk(eps, k=self.neighbors, dim=1)
        batch_size = features.shape[0]
        device = features.device
        cls_w = self.conv_pred_semantic_first.conv_cls.weight.unsqueeze(0).squeeze(-1).repeat(batch_size, 1, 1)
        if self.feature_center_loss:
            cls_w = cls_w + self.feature_center.repeat(batch_size, 1, 1)
        represent = torch.bmm(cls_prob, cls_w)#8*256*128
        using_context = True
        origin_features = features.transpose(2,1)#8*256*128
        if using_context:#设置semantic和context边
            z = self.relation_fc(features).transpose(2, 1)#context边
        else:#semantic边
            semantic_represent = represent.transpose(2,1)#8*128*256
            z = self.relation_fc(semantic_represent).transpose(2, 1)  # 8*256*128
        #计算几何边
        geometric = False
        if geometric:
            eps = compute_l2_distance(xyz)
        else:
            eps = torch.bmm(z, z.transpose(2, 1))
        _, indices = torch.topk(eps, k=self.neighbors, dim=1)
        relation = torch.empty(batch_size, 2, self.neighbors * self.num_proposals, dtype=torch.long).to(device)
        relation[:, 1] = torch.Tensor(list(range(self.num_proposals)) * self.neighbors).unsqueeze(0).repeat(batch_size, 1)
        relation[:, 0] = indices.view(batch_size, -1)
        f = torch.ones_like(z)  # 8*256*128
        for batch_id in range(batch_size):
            relation_ = relation[batch_id]
            feature = represent[batch_id]
            # origin_feature = origin_features[batch_id]
            # f[batch_id], _ = self.GAT((origin_feature, relation_))#设置node
            f[batch_id], _ = self.GAT((feature,relation_ ))
        h = self.sg_conv(f.transpose(2,1))#8*128*256
        new_features = torch.cat([features, h],dim=1)#8*256*256
        new_features = self.relation_mlp(new_features)
        return new_features, cls_scores


class GlobalModule(nn.Module):
    def __init__(self, seed_feature_dim=256, proposal_feature_dim=128, num_proposals=256):
        super().__init__()
        self.seed_feature_dim = seed_feature_dim
        self.proposal_feature_dim = proposal_feature_dim
        self.num_proposals = num_proposals
        self.global_conv = torch.nn.Conv1d(self.seed_feature_dim + self.proposal_feature_dim * 2,
                                           self.proposal_feature_dim, 1)

    def forward(self, seed_features, features, origin_features):
        global_feature_1 = F.max_pool1d(seed_features, kernel_size=seed_features.size(2))
        #print(global_feature_1.shape)
        global_feature_2 = F.max_pool1d(features, kernel_size=features.size(2))
        #print(global_feature_2.shape)
        global_features = torch.cat((global_feature_1, global_feature_2), 1)
        global_features = torch.cat((global_features.expand(features.shape[0],
                                                            self.seed_feature_dim + self.proposal_feature_dim,
                                                            self.num_proposals), features), 1)
        #print(global_features.shape)
        global_features = self.global_conv(global_features)
        global_features = torch.sigmoid(torch.log(torch.abs(global_features)))
        new_features = origin_features * global_features
        return new_features


if __name__=='__main__':
    # gat = GAT(num_of_layers=3,
    #           num_heads_per_layer=[8,8,1],
    #           num_features_per_layer=[128, 128, 128, 128],
    #           add_skip_connection=False,
    #           bias=True,
    #           dropout=0.6,
    #           log_attention_weights=False
    #           ).cuda()
    # GATLayer = GATLayer(num_in_features=128, num_out_features=128, num_of_heads=5).cuda()
    # z = torch.rand(2, 128, 256)
    # z = z.transpose(2, 1)
    # print(z.shape)
    # eps = torch.bmm(z, z.transpose(2, 1))
    # _, indices = torch.topk(eps, k=16, dim=1)
    # represent = torch.randn(2, 256, 128)
    # relation = torch.empty(2, 2, 16 * 256, dtype=torch.long)
    # relation[:, 0] = torch.Tensor(list(range(256)) * 16).unsqueeze(0).repeat(2, 1)
    # relation[:, 1] = indices.view(2, -1)
    # print(relation.shape)
    # for batch in range(2):
    #     relation_ = relation[batch].cuda()
    #     represent_ = represent[batch].cuda()
    #     new_repre, edge_index = gat((represent_, relation_))
    #     print(new_repre.shape)
    a = GraphRelationModule().cuda()
    features = torch.rand(8, 128, 256).cuda()
    x = torch.rand((8, 256, 3)).cuda()
    features, _ = a(features,x)
    print(features.shape)
    x = torch.rand((2,256,3))
    new_x = compute_l2_distance(x)