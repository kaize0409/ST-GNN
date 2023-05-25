import numpy as np

import torch
from tqdm import tqdm
from torch import nn

torch.manual_seed(99)
np.random.seed(99)
torch.cuda.empty_cache()

from data import *

import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

seq_len = 10
device = torch.device('cpu')

task = Tasks('prediction')
data1 = TimeSeriesDataset(task=task,data_path="data/all_avg.npy", seq_length=seq_len, batch_size=1, prediction_window=1)
train_iter, test_iter, nb_features = data1.get_loaders()


graph_matrix = np.load('data/graph.npy')
#np.fill_diagonal(graph_matrix, 1)
edge_index = torch.LongTensor(graph_matrix.nonzero())

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1, bias=True)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        #h = F.relu(h_0)
        h = self.linear(h_0)
        return h, h_0, c_0


model = RecurrentGCN(node_features=seq_len).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



for epoch in range(100):
    model.train()
    cost = 0
    h, c = None, None
    for time, snapshot in enumerate(train_iter):
        feature, target = snapshot
        feature = feature[0].T.to(device)
        target = target[0].T.to(device)
        y_hat, h, c = model(feature, edge_index, None, h, c)
        cost += nn.L1Loss(reduction='mean')(y_hat, target) #torch.mean((y_hat - target) ** 2)
    cost = cost / (time + 1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Training Loss is ", cost.item(), " for epoch ", epoch)

    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_iter):
        feature, target = snapshot
        feature = feature[0].T.to(device)
        target = target[0].T.to(device)
        y_hat, h, c = model(feature, edge_index, None, h, c)
        cost = cost + nn.L1Loss(reduction='mean')(y_hat[983:], target[983:])#torch.mean((y_hat - target) ** 2)
    cost = cost / (time + 1)
    cost = cost.item()
    print('Test MSE', cost, " for epoch ", epoch)
    # print("MSE: {:.4f}".format(cost))

