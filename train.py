from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import dataloader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import random


train_data = dataloader.DCMDatasetLoader_3windows("./data/TrainingData", get_meta=True)

n_train = len(train_data)

split = int(n_train * 0.2)
indices = list(range(n_train))[:]
print(indices[-100:])
random.shuffle(indices)
print(indices[split:])
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

train_dataloader = DataLoader(dataset=train_data, batch_size=16, sampler=train_sampler)
valid_dataloader = DataLoader(dataset=train_data, batch_size=8, sampler=valid_sampler)




# test_data = dataloader.DCMDatasetLoader("./data/TestingData")
# test_dataloader = DataLoader(dataset=test_data, batch_size=4)

for imgs, metadata, lbls in valid_dataloader:
    print('Size of image:', imgs.size())  # batch_size*3*224*224
    print('Type of image:', imgs.dtype)   # float32
    print('Size of label:', lbls.size())  # batch_size
    print('Type of label:', lbls.dtype)   # int64(long)
    print(type(imgs),type(lbls))
    # plt.imshow(imgs[5][0])
    # plt.show()
    break

for imgs, metadata, lbls in train_dataloader:
    print('Size of image:', imgs.size())  # batch_size*3*224*224
    print('Type of image:', imgs.dtype)   # float32
    print('Size of label:', lbls.size())  # batch_size
    print('Type of label:', lbls.dtype)   # int64(long)
    print(type(imgs),type(lbls))
    # plt.imshow(imgs[5][0])
    # plt.show()
    break

for i, (imgs, metadata, lbls) in enumerate(train_dataloader):
    print(lbls)
    if i > 10:
        break

# max = 0
# for i, (imgs, metadata, lbls) in enumerate(train_dataloader):
#     # print(torch.max(metadata))
#     if torch.max(metadata) > max:
#         max = torch.max(metadata)
# print(max) # 1686.6226
print("==")
for i, (imgs, metadata, lbls) in enumerate(valid_dataloader):
    print(lbls)
    if i > 10:
        break

def evaluate(model,dataloader):
    confusion_matrix = np.zeros((6, 6), dtype=float)
    results = []

    net.eval()
    fc_o.eval()
    drop_fc.eval()
    with torch.no_grad():
        acc=[]
        correct = 0.0
        total = 0.0
        running_loss = 0.0
        for inputs, metadata, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels)
            metadata = metadata.to(device)
            # zero the parameter gradients
            res_outputs = net(inputs)
            concat = torch.cat([res_outputs, metadata/1686], 1)
            outputs = fc_o(concat)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()
            for i in range(labels.size(0)):
                confusion_matrix[labels[i]][predicted[i]] += 1

        # Normalized
        for i in range(len(confusion_matrix)):
            confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()
        print(confusion_matrix)
        #print("Accuracy:", 100. * correct / total)

        return 100. * correct / total, running_loss / len(dataloader)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)
#net = EEGNet().to(device)
resume = True
if not resume:
    # net = models.resnext50_32x4d(pretrained=True).to(device)
    net = models.resnet50(pretrained=True).to(device)

    # print(net)
    # net.conv1 = nn.Conv2d(1,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

    fc2 = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048,256)), #256
        ('relu', nn.ReLU())
    ]))
    fc_o = nn.Linear(257, 6).to(device)
    net.fc = fc2.to(device)
    drop_fc = nn.Dropout(0.3)
else:
    print("load old model")
    net = torch.load("17-61.583.pth").to(device)
    fc_o = nn.Linear(257, 6).to(device)
    drop_fc = nn.Dropout(0.3)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5, amsgrad=True)
# optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=5e-5, momentum=0.9, nesterov=True)
optimizer_fc = optim.Adam(list(fc_o.parameters())+list(drop_fc.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5, amsgrad=True)
#optimizer_fc = optim.SGD(list(fc_o.parameters())+list(drop_fc.parameters()), lr=1e-2, weight_decay=5e-5, momentum=0.9, nesterov=True)



sch = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
sch_fc = optim.lr_scheduler.StepLR(optimizer_fc, 2, gamma=0.9)
# sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
epochs = 500

writer = SummaryWriter('log')

acc_train = {}
acc_test = {}
loss_test = {}

for epoch in range(epochs):  # loop over the dataset multiple times
    total = 0.0
    correct = 0.0
    running_loss = 0.0
    for inputs, metadata, labels in train_dataloader:
        net.train()
        fc_o.train()
        drop_fc.train()
        inputs, labels = inputs.to(device), labels.to(device)
        metadata = metadata.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        optimizer_fc.zero_grad()
        res_outputs = net(inputs)
        res_outputs = drop_fc(res_outputs)
        concat = torch.cat([res_outputs, metadata/1686], 1)
        outputs = fc_o(concat)
        # outputs = drop_fc(outputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer_fc.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()
        running_loss += loss.item()

    #sch.step()
    #sch_fc.step()
    print("epoch:", epoch)
    print("Training Loss ", running_loss/len(train_dataloader))
    acc_train[epoch] = 100. * correct / total
    acc_test[epoch], loss_test[epoch] = evaluate(net, valid_dataloader)
    print("Training acc:", acc_train[epoch])
    print("Validation acc:", acc_test[epoch])
    writer.add_scalar('Train/Loss', running_loss/len(train_dataloader), epoch)
    writer.add_scalar('Train/Acc', acc_train[epoch], epoch)
    writer.add_scalar('Val/Acc', acc_test[epoch], epoch)
    writer.add_scalar('Val/Loss', loss_test[epoch], epoch)
    # if ((epoch % 20 == 0) and epoch > 0 and acc_test[epoch] > 95.0) or (epoch == epochs-1):
    if ((epoch % 1 == 0) and epoch > 0 ) or (epoch == epochs - 1):
        torch.save(net,str(epoch)+"-"+str(acc_test[epoch])+".pth")
        torch.save(fc_o,str(epoch)+"-"+str(acc_test[epoch])+"_fc.pth")
        torch.save({
            'epoch': epoch,
            'coder': net.state_dict(),
            'optim': optimizer.state_dict(),
            'scher': sch.state_dict()
        }, "model.ckpt")

plt.plot(list(acc_train),list(acc_train.values()))
plt.plot(list(acc_test),list(acc_test.values()))
plt.show()