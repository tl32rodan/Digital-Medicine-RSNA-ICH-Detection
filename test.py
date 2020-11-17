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
import xlsxwriter

label_list = ["epidural", "healthy", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]

test_data = dataloader.DCMDatasetLoader_3windows_test("./data/TestingData", get_meta = True)

test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

# test_data = dataloader.DCMDatasetLoader("./data/TestingData")
# test_dataloader = DataLoader(dataset=test_data, batch_size=4)

for imgs in test_dataloader:
    print('Size of image:', imgs.size())  # batch_size*3*224*224
    print('Type of image:', imgs.dtype)   # float32
    # plt.imshow(imgs[0][0])
    # plt.show()
    break

def evaluate(model, fc, dataloader):
    results = []
    model.eval()
    fc.eval()
    with torch.no_grad():
        acc=[]
        correct = 0.0
        total = 0.0
        for i, inputs in enumerate(dataloader):
            inputs, metadata = inputs
            inputs = inputs.to(device)
            meta = meta.to(device)

            outputs = model(inputs)
            concat = torch.cat([outputs, metadata / 1686], 1)
            outputs = fc(concat)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            name = "Test_" + "{:03}".format(i+1)

            results.append([name, label_list[predicted]])
        return results

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

net = torch.load("20-62.916666666666664.pth").to(device)
fc = torch.load("20-62.916666666666664_fc.pth").to(device)

output = evaluate(net, fc, test_dataloader)

with xlsxwriter.Workbook('output.xlsx') as workbook:
    worksheet = workbook.add_worksheet()

    for row_num, data in enumerate(output):
        worksheet.write_row(row_num, 0, data)
