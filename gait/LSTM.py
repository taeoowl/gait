import os
import glob
import ast
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# LSTM Module
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, learning_rate):
        super(LSTM, self).__init__() # 상속한 nn.Module에서 RNN에 해당하는 init 실행
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_acc = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm_gyro = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.reg_module1 = nn.Sequential(
            nn.Linear(hidden_size*2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        

    def forward(self, acc, gyro): 
        h0 = torch.zeros(self.num_layers, acc.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, acc.size(0), self.hidden_size).to(device)

        out_acc, _ = self.lstm_acc(acc, (h0, c0))
        out_gyro, _ = self.lstm_gyro(gyro, (h0, c0))

        enc_input = torch.cat((out_acc[:, -1, :], out_gyro[:, -1, :]), dim=1)
        out_lstm = self.reg_module1(enc_input).to(device)
        
        return out_lstm, enc_input