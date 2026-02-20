# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 08:59:58 2026

@author: shams
"""

# CHANGE THE DIRECTORY OF DATA YOU WANT TO TRAIN
FOLDERS = '../Solution/SiouxFalls/Output1'
UNIQUE_PATH_DICT_train = '../Generate_data/SiouxFalls/unique_paths.pkl'
NODE_POSITION = '../Generate_data/SiouxFalls/SiouxFalls_node.csv'

# CHART TITLES
TRAIN_HISTORY_TITLE = "SINGLE CLASS NETWORK - TRAINING HISTORY"
ERROR_TITLE = "SINGLE CLASS NETWORK"

DATA_SIZE = 4000
TRAIN_RATE = 0.7
VAL_RATE = 0.2
TEST_RATE = 0.1
BATCH_SIZE = 64

# TRAINING 
device = 'gpu'
input_dim = 7
output_dim = 3
d_model = 128
heads = 8
E_layer = 8
D_layer = 1
epochs = 500
learning_rate = 0.001
dropout = 0.1
l2_reg = 1e-6