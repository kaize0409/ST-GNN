import numpy as np

import torch
from tqdm import tqdm
from torch import nn

torch.manual_seed(99)
np.random.seed(99)
torch.cuda.empty_cache()

from data import *
from model import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task = Tasks('prediction')
data1 = TimeSeriesDataset(task=task,data_path="data/all_avg.npy", seq_length=10, batch_size=1, prediction_window=1)
train_iter, test_iter, nb_features = data1.get_loaders()

seq_len = 10


model = Encoder(hidden_size_encoder=8, seq_len=seq_len, dropout=0.3, input_size=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



for epoch in range(100):
    model.train()
    total_loss = 0
    h, c = None, None
    for time, snapshot in enumerate(train_iter):
        feature, target = snapshot
        feature = feature.T.to(device)
        target = target.T.to(device)

        output, _, _ = model(feature.to(device))
        target = target.to(device)
        loss = nn.L1Loss(reduction='mean')(output[:,-1,:].reshape((-1,1)), target.reshape((-1,1)).to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print("Total Loss is ", total_loss, " for epoch ", epoch)

    model.eval()
    num_test_batch = 0
    test_loss_total = 0
    for time, snapshot in enumerate(test_iter):
        feature, target = snapshot
        feature = feature.T.to(device)
        target = target.T.to(device)

        output, _, _ = model(feature.to(device))
        target = target.to(device)
        loss = nn.L1Loss(reduction='mean')(output[983:, -1, :].reshape((-1, 1)), target[983:].reshape((-1, 1)).to(device))
        test_loss_total += loss.item()
        num_test_batch += 1
        # print(output[:,-1].reshape(1,-1), target[:, -1].reshape(1,-1))
    test_loss = test_loss_total / num_test_batch
    print('Test MSE', test_loss, " for epoch ", epoch)
