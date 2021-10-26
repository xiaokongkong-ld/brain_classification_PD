import random
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
import os
import heapq
import pandas as pd
from Test_Class import TestMatrix

DATA_PATH = './ABIDE'
groups = ['autism', 'control']


def load_data(group):
    """ read .npz data from disk """
    data_path = os.path.join(DATA_PATH)
    load_name = '/' + group + '.npz'
    data_path = data_path + load_name
    data = np.load(data_path)['a']
    print(f'load data from: {data_path}')
    return data


def get_data(group_list):
    """ read .npz data from disk by group list and put into subject list"""
    sub_list = []
    group_number = len(group_list)
    for i in range(group_number):
        subs_data = load_data(group_list[i])
        count = 0
        for sub in subs_data:
            count += 1
            sub_list.append([sub, i])
    return sub_list


def show_subs_labels(data):
    """ show data and its label"""
    sub_list_len = len(data)
    print(f'we have {sub_list_len} subs')
    print('labels are:')
    for i in range(sub_list_len):
        label = data[i][1]
        group = groups[label]
        print(f'number: {i + 1}, group: {group}')


def matrix_filter_topK(matrix, K):
    """ filt matrix with top K max"""
    m_l = len(matrix[0])
    index = [x for x in range(m_l)]
    f_mat = np.zeros(m_l)

    for x in matrix:
        y = heapq.nlargest(K, range(len(x)), x.take)
        y_else = np.setdiff1d(index, y)
        x[y] = 1
        x[y_else] = 0
        f_mat = np.vstack((f_mat, x))
    f_mat = np.delete(f_mat, np.s_[0], axis=0)
    return f_mat


def create_dataset(data, top_K):
    dataset_list = []
    for i in range(len(data)):
        correlation_matrix_f = matrix_filter_topK(data[i][0], top_K)
        edge_index_coo = coo_matrix(correlation_matrix_f)
        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)

        feature_matrix = data[i][0]
        graph_data = Data(x=torch.tensor(feature_matrix, dtype=torch.float32), edge_index=edge_index_coo,
                          y=torch.tensor(data[i][1]))
        dataset_list.append(graph_data)
    return dataset_list

def train_test_split(data, train_percent):
    sub_train = []
    sub_test = []
    num_sub = len(data)
    node = [x for x in range(num_sub)]
    train_id = random.sample(node, int(train_percent*num_sub))
    for x in train_id:
      node.remove(x)
    test_id = node

    for i in train_id:
        sub_train.append(data[i])
    for j in test_id:
        sub_test.append(data[j])

    return sub_train, sub_test

def show_dataset(dataset, datatype):

    print(f'.....................................{datatype} dataset ............................................')
    for i in range(len(dataset)):
        print(dataset[i])

if __name__=="__main__":

  subs_data = get_data(groups)

  data=create_dataset(subs_data,6)
  show_dataset(data, 'all')