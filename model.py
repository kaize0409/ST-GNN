import numpy as np

import torch


torch.manual_seed(99)
np.random.seed(99)
torch.cuda.empty_cache()
import torch.nn.functional as F


from torch import nn
from torch.autograd import Variable

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


###########################################################################
################################ ENCODERS #################################
###########################################################################

class Encoder(nn.Module):
    def __init__(self, hidden_size_encoder, seq_len, dropout, input_size: int):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size_encoder
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size_encoder,  batch_first=True)
        # self.Linear0 = nn.Linear(input_size, 128, bias=False)
        #self.Linear1 = nn.Linear(hidden_size_encoder, 32, bias=True)
        self.Linear2 = nn.Linear(hidden_size_encoder, 1, bias=True)
        self.dropout = dropout

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        # h_0, c_0 = (init_hidden(input_data, self.hidden_size),
        #             init_hidden(input_data, self.hidden_size))
        output, (hn, cn) = self.lstm(input_data)
        #output = torch.relu(self.Linear1(output))
        #output = F.dropout(output, self.dropout, training=self.training)
        output = self.Linear2(output)
        return output, hn, cn











