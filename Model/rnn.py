from torch.nn import Module
import torch.nn as nn
class RNN(Module):

    def __init__(self, num_in, num_layers, num_hidden, num_out):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(num_in, num_hidden, num_layers, dropout=0.3)
        self.body = nn.Sequential(nn.Linear(num_hidden, num_hidden),nn.Dropout(0.3),nn.Linear(num_hidden, num_out),nn.ReLU())

    def forward(self, X):
        X, _ = self.lstm(X)
        return self.body(X[-1])
