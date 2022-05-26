import torch
import numpy as np
from torch import nn
from models.DCNN import DcnnNet
from models.DCNN_conf import *
from conf import using_data_type


class DCNN_pres_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pres_model = DcnnNet(pres_sensors * 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv1d_output_2*unitstep, dense_output_1)
        self.batchnorm1d = nn.BatchNorm1d(dense_output_1)
        self.fc2 = nn.Linear(dense_output_1, final_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):
        inp = inp[:, :, :pres_sensors * 2]

        inp_transpose = inp.transpose(1, 2).contiguous()

        pres_res = self.pres_model(inp_transpose)

        fc1 = self.fc1(pres_res)
        relu1 = self.relu(fc1)
        batch3 = self.batchnorm1d(relu1)
        dropout1 = self.dropout(batch3)
        fc2 = self.fc2(dropout1)
        relu2 = self.relu(fc2)
        output = self.softmax(relu2)

        return output


class DCNN_multimodal_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pres_model = DcnnNet(pres_sensors * 2)
        self.acc_model = DcnnNet(6)
        self.gyro_model = DcnnNet(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv1d_output_2*unitstep*len(using_data_type), dense_output_1)
        self.batchnorm1d = nn.BatchNorm1d(dense_output_1)
        self.fc2 = nn.Linear(dense_output_1, final_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):

        # inp를 pres, acc, gyro로 나눠준다
        pres_data = inp[:, :, :pres_sensors * 2]
        acc_data = inp[:, :, pres_sensors * 2:(pres_sensors * 2 + 6)]
        gyro_data = inp[:, :, (pres_sensors * 2 + 6):(pres_sensors * 2 + 12)]

        pres_data = pres_data.transpose(1, 2).contiguous()
        acc_data = acc_data.transpose(1, 2).contiguous()
        gyro_data = gyro_data.transpose(1, 2).contiguous()

        if 'pres' in using_data_type:
            pres_res = self.pres_model(pres_data)
        if 'acc' in using_data_type:
            acc_res = self.acc_model(acc_data)
        if 'gyro' in using_data_type:
            gyro_res = self.gyro_model(gyro_data)

        concat = None
        if 'pres' in using_data_type:
            if concat is None:
                concat = pres_res
            else:
                concat = torch.cat([concat, pres_res], dim=1)
        if 'acc' in using_data_type:
            if concat is None:
                concat = acc_res
            else:
                concat = torch.cat([concat, acc_res], dim=1)

        if 'gyro' in using_data_type:
            if concat is None:
                concat = gyro_res
            else:
                concat = torch.cat([concat, gyro_res], dim=1)
        # concat = torch.cat([pres_res, acc_res, gyro_res], dim=1)

        fc1 = self.fc1(concat)
        relu1 = self.relu(fc1)
        batch3 = self.batchnorm1d(relu1)
        dropout1 = self.dropout(batch3)
        fc2 = self.fc2(dropout1)
        relu2 = self.relu(fc2)
        output = self.softmax(relu2)

        return output


def ccc(pres, acc, gyro):
    concat = None
    if 'pres' in using_data_type:
        if concat is None:
            concat = pres
        else:
            concat = torch.cat([concat, pres], dim=1)
    if 'acc' in using_data_type:
        if concat is None:
            concat = acc
        else:
            concat = torch.cat([concat, acc], dim=1)

    if 'gyro' in using_data_type:
        if concat is None:
            concat = gyro
        else:
            concat = torch.cat([concat, gyro], dim=1)
    return concat
