import torch
from torch_geometric.loader import DataLoader
# from model import GCN
from data_process_PD import train_test_split, create_dataset, get_data, show_dataset
from data_process_ABIDE import train_test_split, create_dataset, get_data, show_dataset
from model_new import HGCN_pyg,GCN,GAT,GraphSage
import torch
from torch_geometric.loader import DataLoader
# from PyG_developement_model import GAT, Graphsage, GCN
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

matrix_type = ['FC', 'SC']
datatype = ['timeseries', 'matrix']
# groups = ['PD_PD', 'PD_control', 'PD_Prodromal', 'PD_SWEDD']
# groups = ['PD_PD', 'PD_control', 'PD_Prodromal']
# groups = ['PD_PD', 'PD_control', 'PD_SWEDD']
# groups = ['PD_PD', 'PD_Prodromal']
# groups = ['PD_PD', 'PD_control']
groups = ['autism', 'control']

print("\n---------Starting to load Data---------\n")

## PD dataset
# subs_data_fc = get_data(matrix_type[0], groups)
# subs_data_sc = get_data(matrix_type[1], groups)
# matrix_dim = len(subs_data_sc[0][0])
# print(matrix_dim)
# dataset = create_dataset(subs_data_sc, subs_data_sc, 20)
#
# train_list, test_list = train_test_split(dataset, 0.70)
# show_dataset(dataset, 'original')
# show_dataset(train_list, 'train')
# show_dataset(test_list, 'test')

## ABIDE dataset
subs_data_fc = get_data(groups)

matrix_dim = len(subs_data_fc[0][0])
print(f'Input matrix dimension: {matrix_dim}')
dataset = create_dataset(subs_data_fc, 10)

train_list, test_list = train_test_split(dataset, 0.7)
show_dataset(dataset, 'original')
show_dataset(train_list, 'train')
show_dataset(test_list, 'test')

print("\n-----------Data loaded-----------\n")

train_loader = DataLoader(train_list, batch_size=5, shuffle=True)
test_loader = DataLoader(test_list, batch_size=5, shuffle=True)

group_type=len(groups)

# model = GCN(hidden_channels=64, channel_in=matrix_dim, channel_out=group_type)
model = HGCN_pyg(c=1, hidden_channels=64, channel_in=matrix_dim, channel_out=group_type)
# model = GAT(hidden_channels=64, channel_in=matrix_dim, channel_out=group_type)
# model = GraphSage(hidden_channels=64, channel_in=matrix_dim, channel_out=group_type)

print("Model:\n\t", model)
optimizer = torch.optim.Adam(model.parameters()
                             , lr=0.001
                             , weight_decay=5e-4
                             # , weight_decay=0.001
                             )
# optimizer = torch.optim.SGD(model.parameters()
#                             , lr=0.001
#                             #, momentum=0.9
#                             , weight_decay=0.0001
#                             #, nesterov=True
#                             )
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
criterion = torch.nn.CrossEntropyLoss()

# criterion = F.nll_loss()

def train(model_to_train, train_dataset_loader, loss_function, model_optimizer):
    model_to_train.train()

    for data in train_dataset_loader:  # Iterate in batches over the training dataset.

        out = model_to_train(data.x, data.edge_index
                             , data.batch
                             )  # Perform a single forward pass.
        loss = loss_function(out, data.y)  # Computing the loss.
        # loss = F.nll_loss(out, data.y)
        loss.backward()  # Deriving gradients.
        model_optimizer.step()  # Updating parameters based on gradients.
        model_optimizer.zero_grad()  # Clearing gradients.
    # scheduler.step()

def test(model, loss_function, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        # loss = F.nll_loss(out, data.y)
        loss = loss_function(out, data.y)  # Computing the loss.
    return loss, correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(1, 101):
    train(model, train_loader, criterion, optimizer)
    loss_train, train_acc = test(model, criterion, train_loader)
    loss_test, test_acc = test(model, criterion, test_loader)
    if epoch % 2 == 0:
        #         print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss_train}')
