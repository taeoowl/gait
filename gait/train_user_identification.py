import torch
import numpy as np
import csv

from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as data
from torchsummary import summary

from models.DCNN_model import DCNN_pres_model, DCNN_multimodal_model
from conf import *
from utils.load_gilon import load_gilon
from utils.load_3L import load_3L, load_3L_disease
from train_utils.dataset import *
from models.DCNN_conf import *


#  =============================================================

#      일단 이 파일은 GilOn(9명)과 3L(40명) 데이터에 대한
#      User Identification 작업을 수행하는 train 파일입니다
#          (추후에 기능 통합할 수 있으면 통합할 예정)
#      아직 모델을 저장하는 기능은 만들지 않았어요.

#      conf.py에서 train.py쪽과 DCNN_conf.py에서 unitstep을 바꿔서 사용하세요
#      보고 알아서 생각하시기를

#  =============================================================


def main(step):

    if data_source == 'GilOn':
        # get data from .pkl file
        x_train, y_train, x_val, y_val, x_test, y_test = load_gilon(step)
    elif data_source == '3L':
        x_train, x_val, x_test, y_train, y_val, y_test = load_3L()
    elif data_source == '3L_disease':
        x_train, x_val, x_test, y_train, y_val, y_test = load_3L_disease()
    else:
        print("invalid data_source!")
        print("need proper data for User Identification! ex) GilOn, 3L")
        exit()

    if data_source == 'GilOn' or data_source == '3L' or data_source == '3L_disease':
        train_dataset = GilOnDataset(x_train, y_train)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
        valid_dataset = GilOnDataset(x_val, y_val)
        valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=256, shuffle=True)
        test_dataset = GilOnDataset(x_test, y_test)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

    print("GPU available:", torch.cuda.is_available(), " GPU_name:", torch.cuda.get_device_name())
    device = torch.device('cuda')

    if using_data_type == ['pres']:
        model = DCNN_pres_model()
    else:
        model = DCNN_multimodal_model()
    model = model.cuda()
    summary(model, input_size=x_test[0].shape)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

    for epoch in range(epochs + 1):

        # train
        model.train()
        for batch_ind, samples in enumerate(train_loader):
            train_accuracy = 0.0

            # print("batch index:" + str(batch_ind), " samples:" + str(samples))
            x_t, y_t = samples
            x_t, y_t = x_t.to(device), y_t.to(device)

            # predict
            pred = model(x_t)

            # cost 계산
            cost = F.cross_entropy(pred, y_t)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            _, predicted = torch.max(pred, 1)
            train_accuracy += (predicted == y_t).sum().item()
            train_accuracy = 100 * (train_accuracy/y_t.size(0))

            # print(f"Batch {batch_ind+1}/{len(train_loader)} Train_accuracy:{train_accuracy:.3f}")

        running_val_loss = 0.0
        running_test_loss = 0.0
        val_accuracy = 0.0
        test_accuracy = 0.0
        total = 0

        with torch.no_grad():
            # validation accuracy 계산
            model.eval()
            for samples in valid_loader:
                x_t, y_t = samples
                x_t, y_t = x_t.to(device), y_t.to(device)
                pred = model(x_t)
                val_loss = F.cross_entropy(pred, y_t)

                # 가장 확률이 높은걸 고름
                _, predicted = torch.max(pred, 1)
                running_val_loss += val_loss.item()
                total += y_t.size(0)
                val_accuracy += (predicted == y_t).sum().item()

        valid_loss = running_val_loss/len(valid_loader)
        accuracy = (100 * val_accuracy/total)
        # end of one experiment
        print(f"Epoch {epoch}/{epochs} Train_accuracy:{train_accuracy:.3f} Valid_loss:{valid_loss:.3f} Valid_Accuracy:{accuracy:.3f}")

    # for test data
    with torch.no_grad():
        total = 0
        # test accuracy 계산
        for samples in test_loader:
            x_t, y_t = samples
            x_t, y_t = x_t.to(device), y_t.to(device)
            pred = model(x_t)
            val_loss = F.cross_entropy(pred, y_t)

            # 가장 확률이 높은걸 고름
            _, predicted = torch.max(pred, 1)
            running_test_loss += val_loss.item()
            total += y_t.size(0)
            test_accuracy += (predicted == y_t).sum().item()

    test_loss = running_test_loss/len(valid_loader)
    accuracy = (100 * test_accuracy/total)
    # write in csv file
    print(f"Test_loss:{test_loss:.3f} Test_Accuracy:{accuracy:.3f}")

    torch.cuda.empty_cache()

    return accuracy


result = dict()

for step in steps:
    result[step] = list()
    for i in range(repeats):
        accuracy = main(step)
        result[step].append(accuracy)

# write in csv file
file_dir = './result/' + result_dir
f = open(file_dir, 'w', encoding='utf-8')
wr = csv.writer(f)
for step in result:
    step_data = result[step]
    wr.writerow([step] + step_data)


print('end')

