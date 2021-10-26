import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GAE,SAGEConv
from torch_geometric.nn import global_mean_pool

# import manifolds
# from layers.hyp_layers import HyperbolicGraphConvolution, HypLinear, HypAgg, HypAct
# from models.encoders import HGCN
#
# #######################hyperbolic graph coonvolutional neural network###########################
# class HGCN(torch.nn.Module):
#     def __init__(self, c, args):
#         super(HGCN, self).__init__(c)
#         self.manifold = getattr(manifolds, args.manifold)()
#         assert args.num_layers > 1
#         dims = 128
#         acts = 'relu'
#         self.curvatures = 1
#         #        self.curvatures.append(self.c)
#         hgc_layers = []
#         for i in range(2):
#             c_in, c_out = self.curvatures, self.curvatures
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             hgc_layers.append(
#                 hyp_layers.HyperbolicGraphConvolution(
#                     self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
#                     args.local_agg
#                 )
#             )
#         self.layers = torch.nn.Sequential(*hgc_layers)
#         self.encode_graph = True

#######################hyperbolic graph coonvolutional neural network###########################

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(32, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(32,2)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        x = self.lin1(x, edge_index)
        x = F.relu(x)
        x = self.lin2(x, edge_index)
        x = F.relu(x)
        x = self.lin3(x, edge_index)

        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, indim, outdim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(indim, 84)
        self.conv2 = GCNConv(84, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 32)
        self.lin1 = Linear(32,outdim)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)

        return x

class GAT(torch.nn.Module):
    def __init__(self,hidden_channels,indim, outdim):
        super(GAT,self).__init__()
        self.gat1 = GATConv(indim, hidden_channels,dropout = 0.2)
        self.gat2 = GATConv(hidden_channels, 64,dropout = 0.2)
        self.lin1 = Linear(64, outdim)
        self.dropout = torch.nn.Dropout(p=0.2)
    def forward(self,x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)

        return x

class Graphsage(torch.nn.Module):
    def __init__(self, hidden_channels,indim, outdim):
        super(Graphsage, self).__init__()
        self.sage1 = SAGEConv(indim, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, 64)
        self.lin1 = Linear(64, outdim)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.dropout(x)
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        x = F.relu(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x)

        return x


