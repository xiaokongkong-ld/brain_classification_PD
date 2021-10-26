import numpy as np
import heapq
from scipy.sparse import coo_matrix
import random
import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from Test_Class import TestMatrix
from data_process_PD import matrix_filter_topK
from Test_Class import TestMatrix

mat = TestMatrix((4,5),20)
mat2 = TestMatrix((3,6),20)

print(mat.getmat())
print(mat2.getmat())
fil_mat = matrix_filter_topK(mat.getmat(), 2)
fil_mat2 = matrix_filter_topK(mat2.getmat(), 2)
print(fil_mat)
print(fil_mat2)

