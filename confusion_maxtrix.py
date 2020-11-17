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


train_data = dataloader.DCMDatasetLoader_3windows("./data/TrainingData")

# train_dataloader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

n_train = len(train_data)

split = int(n_train * 0.2)
indices = list(range(n_train))
random.shuffle(indices)
print(indices[:])
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
# print(valid_sampler.indices)
train_dataloader = DataLoader(dataset=train_data, batch_size=64, sampler=train_sampler)
valid_dataloader = DataLoader(dataset=train_data, batch_size=64, sampler=valid_sampler)

# test_data = dataloader.DCMDatasetLoader("./data/TestingData")
# test_dataloader = DataLoader(dataset=test_data, batch_size=4)

for imgs, lbls in train_dataloader:
    print('Size of image:', imgs.size())  # batch_size*3*224*224
    print('Type of image:', imgs.dtype)   # float32
    print('Size of label:', lbls.size())  # batch_size
    print('Type of label:', lbls.dtype)   # int64(long)
    print(type(imgs),type(lbls))
    plt.imshow(imgs[0][0])
    plt.show()
    break

for i, (imgs, lbls) in enumerate(train_dataloader):
    print(lbls)
    if i > 10:
        break

print("==")

for i, (imgs, lbls) in enumerate(valid_dataloader):
    print(lbls)
    if i > 10:
        break

# exit()


def evaluate(model,dataloader):
    confusion_matrix = np.zeros((6,6),dtype=float)
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i in range(labels.size(0)):
                confusion_matrix[labels[i]][predicted[i]] += 1
            correct += predicted.eq(labels.data).cpu().sum().item()

        #print(confusion_matrix)
        print("Testing Accuracy:", 100. * correct / total)

        return confusion_matrix

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

net = torch.load("20-62.916666666666664.pth").to(device)
# net = torch.load("149-98.91666666666667.pth").to(device)


confusion_matrix = evaluate(net, valid_dataloader)
print(confusion_matrix)

# Normalized
for i in range(len(confusion_matrix)):
    confusion_matrix[i] = confusion_matrix[i]/confusion_matrix[i].sum()

fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(confusion_matrix)))
ax.set_yticks(np.arange(len(confusion_matrix)))
# ... and label them with the respective list entries
ax.set_xticklabels(np.arange(5))
ax.set_yticklabels(np.arange(5))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
        text = ax.text(j, i,  '{:.2f}'.format(confusion_matrix[i, j]),
                       ha="center", va="center", color="w")

fig.text(0.5, 0.007, 'Predicted label', ha='center')
fig.text(0.08, 0.5, 'True label', va='center', rotation='vertical')

ax.set_title("Normalized confusion matrix")
fig.tight_layout()
plt.colorbar(im)
plt.show()