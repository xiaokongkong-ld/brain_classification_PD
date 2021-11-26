import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from hgcn_conv import HGCNConv,HypAct
import torch
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import manifolds
import layers.hyp_layers_new as hyp_layers
import numpy as np
from layers.att_layers import GraphAttentionLayer
import utils_hpy.math_utils as pmath
from torch.nn import Linear
from hgcn_conv import HypLinear

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GCN, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = GCNConv(self.channel_in, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, self.channel_out)
        self.lin1 = Linear(hidden_channels, self.channel_out)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        # return F.log_softmax(x)
        return x

class HGCN_pyg(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """
    def __init__(self, c, hidden_channels, channel_in, channel_out):
        super(HGCN_pyg, self).__init__()
        torch.manual_seed(1234567)
        self.c = c
        self.channel_in = channel_in
        self.channel_out = channel_out
        # self.manifold = getattr(manifolds, 'Euclidean')()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # self.manifold = getattr(manifolds, 'PoincareBall')()
        act = getattr(F, 'relu')
        self.hconv1 = HGCNConv(self.manifold, self.channel_in, hidden_channels, self.c)
        self.hconv2 = HGCNConv(self.manifold, hidden_channels, self.channel_out, self.c)
        self.hyp_act = HypAct(self.manifold, self.c, act)
        self.lin1 = HypLinear(self.manifold, hidden_channels, self.channel_out, self.c, 0.5, True)
        self.lin2 = Linear(hidden_channels, self.channel_out)


    def forward(self, x, edge_index, batch):

        # x = self.manifold.proj_tan0(x, self.c)
        # x = self.manifold.expmap0(x, self.c)
        # x = self.manifold.proj(x, self.c)

        x = self.hconv1(x, edge_index)
        x = self.hyp_act(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.hconv2(x, edge_index)

        x = self.manifold.logmap0(x, c=self.c)
        x = self.manifold.proj_tan0(x, c=self.c)

        # x = F.log_softmax(x)
        x = global_mean_pool(x, batch)
        # return x
        return F.log_softmax(x)

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(1000, 600)
        self.lin2 = Linear(600, 200)
        self.lin3 = Linear(200, 86)
        self.lin4 = Linear(86, 16)
        self.lin5 = Linear(16, 7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin5(x)
        return x
# class HGCN(torch.nn.Module):
#     """
#     Hyperbolic-GCN.
#     """
#
#     def __init__(self, c):
#         super(HGCN, self).__init__()
#         self.c = c
#         # self.manifold = getattr(manifolds, 'Euclidean')()
#         self.manifold = getattr(manifolds, 'Hyperboloid')()
#         # self.manifold = getattr(manifolds, 'PoincareBall')()
#         act = getattr(F, 'relu')
#         # self.lin = Linear(64, 7, dropout=0.0, act=act, use_bias=True)
#         self.hgcov1 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 1433, 100, 1, 1, 0.0, act, 1, 0, 0)
#         self.hgcov2 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 100, 7, 1, 1, 0.0, act, 1, 0, 0)
#         self.encode_graph = True
#
#     def forward(self, x, adj):
#         # print('......................xtan............................')
#
#         x_tan = self.manifold.proj_tan0(x, 1)
#
#         x_hyp = self.manifold.expmap0(x_tan, 1)
#         # print(x_hyp)
#         x_hyp = self.manifold.proj(x_hyp, 1)
#         h = self.hgcov1.forward(x_hyp, adj)
#         h = self.hgcov2.forward(h, adj)
#
#         h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)
#         # h = self.lin(h)
#         return F.log_softmax(h)
#         # return h

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GAT, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = GATConv(self.channel_in, hidden_channels)
        self.conv2 = GATConv(hidden_channels, self.channel_out)
        self.lin1 = Linear(hidden_channels, self.channel_out)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        return F.log_softmax(x)
        # return x

class GraphSage(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GraphSage, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = SAGEConv(self.channel_in, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, self.channel_out)
        self.lin1 = Linear(hidden_channels, self.channel_out)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        return F.log_softmax(x)
        # return x