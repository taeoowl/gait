import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import os 
import glob
import cv2
import itertools



class Gait_Dataset_Salted(Dataset):
    def __init__(self, file_path, bias=False):
        self.bias = bias
        self.file_path = file_path
        self.inputs_acc, self.inputs_gyr, self.stride_length = get_sensor_salted(file_path, bias=bias)
#         self.inputs_spd = get_speed_salted(file_path)
        self.inputs_pst = get_position_salted(file_path, distance=True, bias=bias)
        self.inputs_var = get_variance_salted(file_path)
        
    def __len__(self) :
        return len(self.inputs_acc)
    
    def __getitem__(self, idx):
        return self.inputs_acc[idx], self.inputs_gyr[idx], self.stride_length[idx], self.inputs_pst[idx], self.inputs_var[idx]
    
    
class Gait_Dataset_Axis_Salted(Dataset):
    def __init__(self, file_path, axis=None):
        self.file_path = file_path
        self.inputs_x, self.inputs_y, self.inputs_z, self.stride_length = get_axis_sensor_salted(file_path)
        self.inputs_pst = get_position_salted(file_path, distance=True)
        
    def __len__(self) :
        return len(self.stride_length)
    
    def __getitem__(self, idx):
        return self.inputs_x[idx], self.inputs_y[idx], self.inputs_z[idx], self.stride_length[idx], self.inputs_pst[idx]



def get_event_salted(file_path):
    
    df = pd.read_csv(file_path, skiprows=2)
    
    # 압력값 불러오기
    prs = df.filter(regex="R_value")
    prs_mean = np.mean(prs.iloc[:, 0:4], axis=1) 
    
    # Gaussian Filter : Smoothing pressure data
    prs_mean_gf = pd.Series(scipy.ndimage.gaussian_filter1d(prs_mean, 2))
    
    # Level Shift 
    lev_idx = prs_mean_gf > 2 # 가우시안 필터로 인해 0이 아닌 값으로 threshold 설정
    prs_lev = lev_idx.map(lambda x : 1 if x else 0)
    
    # HS, TO Index 추출
    event_idx = []
    for i in range(len(prs_lev)-1):
        if (prs_lev[i] - prs_lev[i+1]) != 0:
            event_idx.append(i)
            
    # HS 이벤트가 먼저 시작하도록 : 나중에 lev shift를 통해 구하는 방법으로 변경하는 것도 괜찮을 듯
    if np.diff(event_idx)[0] < np.diff(event_idx)[1]: # Swing Phase가 Stand Phase보다 길다는 것을 가정 : 장애 데이터에서도 확인 필요
        del event_idx[0]
    
    
    return event_idx

def get_sensor_salted(file_path, normalization=True, bias=False):
    inputs_acc = []
    inputs_gyr = []
    stride_length = []
    for file_name in glob.glob(file_path):
        df = pd.read_csv(file_name, skiprows=2)
        acc = df.filter(regex="R_ACC")
        gyr = df.filter(regex="R_GYRO")
        event_idx = get_event_salted(file_name)

        # HS 이벤트 추출
        event_hs = event_idx[0::2]

        # m/s^2 단위 변환
        acc = (acc / 1000) * 9.8066
        gyr = gyr / 10
        
        # x축과 z축에 적용되는 bias 제거(z축의 경우 중력가속도)
        if bias == False:
            acc['R_ACC_X']= acc['R_ACC_X'] - np.mean(acc['R_ACC_X'])
            acc['R_ACC_Z']= acc['R_ACC_Z'] - np.mean(acc['R_ACC_Z']) 
        
        # Normalization
        if normalization == True:
            scaler = MinMaxScaler()
            acc_norm = scaler.fit_transform(acc)
            gyr_norm = scaler.fit_transform(gyr)

        # 가속도와 자이로 센서 값
        for i in range(1, len(event_hs)):
            if normalization == True:
                inputs_acc.append(np.transpose(cv2.resize(acc_norm[event_hs[i-1]:event_hs[i]], dsize=(3, 300))))
                inputs_gyr.append(np.transpose(cv2.resize(gyr_norm[event_hs[i-1]:event_hs[i]], dsize=(3, 300))))
            else:
                inputs_acc.append(np.transpose(acc[event_hs[i-1]:event_hs[i]]))
                inputs_gyr.append(np.transpose(gyr[event_hs[i-1]:event_hs[i]]))

            #             inputs_gyr.append(acc_norm[event_hs[i-1]:event_hs[i]])
    
        if '3km' in file_name:
            stride_length.append(np.diff(event_hs) * (3000/3600))
        elif '4km' in file_name:
            stride_length.append(np.diff(event_hs) * (4000/3600))
        else:
            stride_length.append(np.diff(event_hs) * (5000/3600))
            
    stride_length = np.round(np.array(list(itertools.chain.from_iterable(stride_length))), 3)
    
    return inputs_acc, inputs_gyr, stride_length

def get_axis_sensor_salted(file_path, normalization=True, bias=False):
    inputs_x = []
    inputs_y = []
    inputs_z = []
    stride_length = []
    for file_name in glob.glob(file_path):
        df = pd.read_csv(file_name, skiprows=2)
        acc = df.filter(regex="R_ACC")
        gyr = df.filter(regex="R_GYRO")
        event_idx = get_event_salted(file_name)

        # HS 이벤트 추출
        event_hs = event_idx[0::2]

        # m/s^2 단위 변환
        acc = (acc / 1000) * 9.8066
        
        # x축과 z축에 적용되는 bias 제거(z축의 경우 중력가속도)
        if bias == False:
            acc['R_ACC_X']= acc['R_ACC_X'] - np.mean(acc['R_ACC_X'])
            acc['R_ACC_Z']= acc['R_ACC_Z'] - np.mean(acc['R_ACC_Z']) 

        # Normalization
        if normalization == True:
            scaler = MinMaxScaler()
            acc_norm = scaler.fit_transform(acc)
            gyr_norm = scaler.fit_transform(gyr)

            
        # 가속도와 자이로 센서 값
        for i in range(1, len(event_hs)):
            if normalization == True:
                inputs_x.append(np.vstack([np.transpose(cv2.resize(acc_norm[:, 0][event_hs[i-1]:event_hs[i]], dsize=(1, 300))), 
                                           np.transpose(cv2.resize(gyr_norm[:, 0][event_hs[i-1]:event_hs[i]], dsize=(1, 300)))]))
                inputs_y.append(np.vstack([np.transpose(cv2.resize(acc_norm[:, 1][event_hs[i-1]:event_hs[i]], dsize=(1, 300))), 
                                           np.transpose(cv2.resize(gyr_norm[:, 1][event_hs[i-1]:event_hs[i]], dsize=(1, 300)))]))
                inputs_z.append(np.vstack([np.transpose(cv2.resize(acc_norm[:, 2][event_hs[i-1]:event_hs[i]], dsize=(1, 300))), 
                                           np.transpose(cv2.resize(gyr_norm[:, 2][event_hs[i-1]:event_hs[i]], dsize=(1, 300)))]))
            else:
                inputs_acc.append(np.transpose(acc[event_hs[i-1]:event_hs[i]]))
                inputs_gyr.append(np.transpose(gyr[event_hs[i-1]:event_hs[i]]))

    
        if '3km' in file_name:
            stride_length.append(np.diff(event_hs) * (3000/3600))
        elif '4km' in file_name:
            stride_length.append(np.diff(event_hs) * (4000/3600))
        else:
            stride_length.append(np.diff(event_hs) * (5000/3600))
            
    stride_length = np.round(np.array(list(itertools.chain.from_iterable(stride_length))), 3)
    
    return inputs_x, inputs_y, inputs_z, stride_length



def get_speed_salted(file_path, bias=False):
    bias = bias
    inputs_acc, _, _ = get_sensor_salted(file_path, normalization=False, bias=bias)
    inputs_spd = []
    for i in range(len(inputs_acc)):
        spd = pd.DataFrame(scipy.integrate.cumulative_trapezoid(inputs_acc[i], dx=(1/100)))
        inputs_spd.append(spd) #m/s
    return inputs_spd

def get_position_salted(file_path, distance=False, bias=False):
    bias = bias
    inputs_spd = get_speed_salted(file_path, bias = bias)
    inputs_pst = []
    for i in range(len(inputs_spd)):
        pst = pd.DataFrame(scipy.integrate.cumulative_trapezoid(inputs_spd[i], dx=(1/100)))
        if distance==True:
            pst = np.array(np.sum(pst, axis=1))
        inputs_pst.append(pst)
    
    scaler = MinMaxScaler()
    inputs_pst = scaler.fit_transform(inputs_pst)

        
    return inputs_pst


def get_variance_salted(file_path):
    inputs_acc, _, _ = get_sensor_salted(file_path, normalization=False)
    inputs_var = []
    for i in range(len(inputs_acc)):
        var = np.round(np.var(inputs_acc[i], axis=1), 3)
        inputs_var.append(var.R_ACC_Y)
        
    scaler = MinMaxScaler()
    inputs_var = scaler.fit_transform(np.array(inputs_var).reshape(-1, 1))
        
    return inputs_var
