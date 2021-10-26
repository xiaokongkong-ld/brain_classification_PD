import torch
from torch_geometric.loader import DataLoader
# from model import GCN
from data_process_PD import train_test_split, create_dataset, get_data, show_dataset
# from data_process_ABIDE import train_test_split, create_dataset, get_data, show_dataset
import torch
from torch_geometric.loader import DataLoader
# from model import GCN
from PyG_developement_model import GAT, Graphsage, GCN

matrix_type = ['FC', 'SC']
datatype = ['timeseries', 'matrix']
groups = ['PD_PD', 'PD_control', 'PD_Prodromal', 'PD_SWEDD']
# groups = ['PD_PD', 'PD_control', 'PD_Prodromal']
# groups = ['PD_PD', 'PD_Prodromal']
# groups = ['PD_PD', 'PD_control']
# groups = ['autism', 'control']

print("\n---------Starting to load Data---------\n")


# subs_data_fc = get_data(matrix_type[0], groups)
subs_data_sc = get_data(matrix_type[1], groups)
matrix_dim = len(subs_data_sc[0][0])
print(matrix_dim)
dataset = create_dataset(subs_data_sc, subs_data_sc, 20)

train_list, test_list = train_test_split(dataset, 0.70)
show_dataset(dataset, 'original')
show_dataset(train_list, 'train')
show_dataset(test_list, 'test')

# subs_data_fc = get_data(groups)
#
# matrix_dim = len(subs_data_fc[0][0])
# print(matrix_dim)
# dataset = create_dataset(subs_data_fc, 17)
#
# train_list, test_list = train_test_split(dataset, 0.7)
# show_dataset(dataset, 'original')
# show_dataset(train_list, 'train')
# show_dataset(test_list, 'test')

print("\n-----------Data loaded-----------\n")

train_loader = DataLoader(train_list, batch_size=5, shuffle=True)
test_loader = DataLoader(test_list, batch_size=5, shuffle=True)

# model = GCN(hidden_channels=64, indim=matrix_dim, outdim=len(groups))
# model = GAT(hidden_channels=64,indim=matrix_dim, outdim=len(groups))
model = Graphsage(hidden_channels=64,indim=matrix_dim, outdim=len(groups))

print("Model:\n\t", model)

optimizer = torch.optim.Adam(model.parameters()
                             , lr=0.0001
                             , weight_decay=0.00001
                             )
criterion = torch.nn.CrossEntropyLoss()


def train(model_to_train, train_dataset_loader, loss_function, model_optimizer):
    model_to_train.train()

    for data in train_dataset_loader:  # Iterate in batches over the training dataset.
        out = model_to_train(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = loss_function(out, data.y)  # Computing the loss.
        loss.backward()  # Deriving gradients.
        model_optimizer.step()  # Updating parameters based on gradients.
        model_optimizer.zero_grad()  # Clearing gradients.


def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 1001):
    train(model, train_loader, criterion, optimizer)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    if epoch % 5 == 0:
        #         print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
