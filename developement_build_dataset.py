import numpy as np
import torch
import torch_geometric
import heapq
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import random
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from torch_geometric.utils import to_networkx
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp

datapath='./developement/'

def matrix_filter_percent(matrix, percent):
    matrix_filter=matrix.copy()
    matrix_filter[matrix>=percent]=1
    matrix_filter[matrix<percent]=0
    return matrix_filter
"""
m=np.array([[0.3,1,0.2,1,4],[9,3,1,2,7],[0.2,5,10,1,2],[2,1,8,7,9]])
m=matrix_filter(m,5)
print(m)
coo = coo_matrix(m)
print(coo)
coo=torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
print(coo)
"""
def matrix_filter_topK(matrix, K):
    m_l=len(matrix[0])
    index=[x for x in range(m_l)]
    f_mat=np.zeros(m_l)

    for x in matrix:
        y=heapq.nlargest(K, range(len(x)), x.take)
        y_else=np.setdiff1d(index, y)
        x[y]=1
        x[y_else]=0
        f_mat = np.vstack((f_mat, x))
    f_mat=np.delete(f_mat, np.s_[0], axis=0)
    return f_mat
"""
m = np.array([[0.3, 1, 0.2, 1, 4], [9, 3, 1, 2, 7], [0.3, 1, 0.2, 1, 4], [3, 5, 10, 1, 2], [2, 1, 8, 7, 9],[2, 1, 8, 7, 9]])
print(m)
f=matrix_filter_topK(m, 2)
print(f)

a = np.random.rand(20, 20)
yi=np.array([1,1,0])
correlation = np.tril(a) + np.tril(a, -1).T
print(correlation)
m=matrix_filter_topK(correlation,4)
print(m)
edge_index_coo = coo_matrix(m)
edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)
print(edge_index_coo)
graph_data = Data(x=torch.tensor(correlation, dtype=torch.float32), edge_index=edge_index_coo,
                          y=torch.tensor(yi).long())
G = to_networkx(graph_data, to_undirected=True)
nx.draw(G)
plt.show()
"""

def train_test_split(data, train_percent, labels):
    sub_train = []
    sub_test = []
    num_sub = len(data)
    node = [x for x in range(num_sub)]
    train_id = random.sample(node, int(train_percent*num_sub))
    for x in train_id:
      node.remove(x)
    test_id = node

    for i in train_id:
        sub_train.append([np.array(data[i])
                             , labels[i]])
    for j in test_id:
        sub_test.append([np.array(data[j]), labels[j]])

    return sub_train, sub_test

def get_data_from_disk(path, num_sub):
    sub_list = []
    for i in range(num_sub):
        #feature_matrix = np.load('./ABIDE/embedding/embeddings' + str(i) + '.npy')
        feature_matrix = np.genfromtxt('./ABIDE/Features_data ' + f'{i:03d}.txt')
        #feature_matrix = np.loadtxt('')
        sub_list.append(feature_matrix)

    labels = np.load(path + 'labelsSEX54.npy')[:num_sub]
    #labels = np.load(path+'labelsDX_GROUP54.npy')[:num_sub]
    #labels = np.load(path + 'labelsChild_Adult155.npy')[:num_sub]
    #labels = np.load(path + 'labelsAgeGroup155.npy')[:num_sub]
    #labels = np.load(path + 'labelsAgeGroup155.npy')[:num_sub]
    return sub_list,labels

# Using the PyTorch Geometric's Data class to load the data into the Data class needed to create the dataset
def create_dataset(data):
    dataset_list = []
    for i in range(len(data)):
        feature = data[i][0]

        # initialize correlation measure, set to vectorize
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([feature])[0]
        #sp.save_npz('./ABIDE/coo_matrix' + str(i) + '.npz', sp.csc_matrix(correlation_matrix))

        feature_matrix=correlation_matrix.copy()
        #feature_matrix=np.load('./ABIDE/embedding/embeddings'+str(i)+'.npy')
        np.fill_diagonal(correlation_matrix, 0)
        correlation_matrix_f=matrix_filter_topK(correlation_matrix,6)
        np.save("./ABIDE/matrix"+ str(i), correlation_matrix_f)
        edge_index_coo = coo_matrix(correlation_matrix_f)
        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)),dtype=torch.long)
        #edge_index_coo = np.vstack((edge_index_coo.row, edge_index_coo.col))
        #coo=pd.DataFrame(edge_index_coo.T)
        #coo.to_csv("./ABIDE/edge"+ str(i) + '.csv', index=False)


        graph_data = Data(x=torch.tensor(feature_matrix, dtype=torch.float32), edge_index=edge_index_coo,
                          y=torch.tensor(data[i][1]).long())
        dataset_list.append(graph_data)
    return dataset_list

datas,label=get_data_from_disk('./ABIDE/', 54)
print(label)
sub_data = []
for sub,la in zip(datas,label):
    sub_data.append([np.array(sub),la])

list_data=create_dataset(sub_data)

"""

sub_data,label = get_data_from_disk(datapath, 10)
train_data, test_data = train_test_split(sub_data,0.9,label)
test_dataset=create_dataset(test_data)
feature = test_data[0][0]

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([feature])[0]
m=matrix_filter_topK(correlation_matrix,6)
#m=matrix_filter_percent(correlation_matrix,0.1)
#m=pd.DataFrame(correlation_matrix)
print(m)

edge_index_coo = coo_matrix(m)
edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)
print(edge_index_coo)
graph_data = Data(x=torch.tensor(correlation_matrix, dtype=torch.float32), edge_index=edge_index_coo,
                          )
G = to_networkx(graph_data, to_undirected=True)
nx.draw(G,with_labels = True)
plt.show()


graph_data = Data(x=torch.tensor(correlation, dtype=torch.float32), edge_index=edge_index_coo,
                          y=torch.tensor(yi).long())
G = to_networkx(graph_data, to_undirected=True)
nx.draw(G)
plt.show()
"""



"""
sub_data,label = get_data_from_disk(datapath, 10)
train_data, test_data = train_test_split(sub_data,0.9,label)
test_dataset=create_dataset(test_data)
feature = test_data[0][0]

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([feature])[0]
correlation_matrix=matrix_filter_topK(correlation_matrix,2)
m=pd.DataFrame(correlation_matrix)
print(m)


sub_data,label = get_data_from_disk(datapath, 30)
train_data, test_data = train_test_split(sub_data,0.8,label)
for i in range(len(train_data)):
    print(train_data[i][1])

for i in range(len(test_data)):
    print(test_data[i][1])

train_dataset=create_dataset(train_data)
test_dataset=create_dataset(test_data)

print(len(test_dataset))
print(len(train_dataset))

#data=graph[0]
#G = to_networkx(data, to_undirected=True)
#nx.draw(G)
#plt.show()

print('====================')
print(f'Number of graphs: {len(graph)}')

print('====================')
# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(data.x.shape)
"""

