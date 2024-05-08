
from input import ChainNetDataset, ProcessData, CollateBatch
from ChainNet import Net
from utils import SaveBestModel
from utils import draw_curve

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import time

def train(model, criterion, optimizer, train_loader, epoch):
    model.train()

    num_batch = 1
    total_loss = 0
    total_paths = 0
    for data in train_loader:
        # zero gradients
        optimizer.zero_grad()

        start_batch = time.time()
        # forward pass and loss
        out = model(data)
        real = data[14].unsqueeze(0)
        loss = criterion(out, real)
        # backward pass
        loss.backward()
        # updates
        optimizer.step()
        end_batch = time.time()

        print(f'Epoch:{epoch+1}, Batch:{num_batch}, Time:{end_batch-start_batch}')
        # total loss
        num_batch = num_batch+1
        total_loss += float(loss)*len(data[5])*2 
        # total execution sequences
        total_paths += len(data[5])
    return total_loss / (total_paths*2) 

@torch.no_grad()
def val(model, criterion, val_loader):
    model.eval()

    total_loss = 0
    total_PError = 0
    total_paths = 0
    for data in val_loader:

        # forward pass
        pred = model(data)
        real = data[14].unsqueeze(0)
        # total percentage error
        PError = abs(real-pred)/(real+1e-5)
        PError = torch.sum(PError)
        total_PError += float(PError)
        # total loss
        loss = criterion(pred, real)
        total_loss += float(loss)*len(data[5])*2 
        # total execution sequences
        total_paths += len(data[5])
    return total_loss / (total_paths*2), total_PError / (total_paths*2) 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 200

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['val'] = []
x_epoch = []

start_data = time.time()
data_list = ProcessData(root='/your folder to save training data', numSamples=40000)
train_dataset = ChainNetDataset(data_list)
data_list = ProcessData(root='/your folder to save validation data', numSamples=10000)
val_dataset = ChainNetDataset(data_list)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=CollateBatch)
val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=CollateBatch)
end_data = time.time()
print(f'Data loading time:{end_data-start_data}')

model = Net(t=8, dim_node=64, dim_path=64, dim_linear=64, num_heads=2, negative_slope=0.2, dropout=0.0)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
LRscheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# initialize SaveBestModel class
save_best_model = SaveBestModel()

for epoch in range(num_epochs):
    x_epoch.append(epoch+1)
    start_epoch = time.time()
    train_loss = train(model, criterion, optimizer, train_loader, epoch)
    end_epoch = time.time()
    y_loss['train'].append(train_loss)
    val_loss, val_PError = val(model, criterion, val_loader)
    y_loss['val'].append(val_loss)
    y_err['val'].append(val_PError)
    LRscheduler.step()

    minPError = save_best_model(val_PError, epoch, model, criterion, optimizer)

    print(f'Epoch:{epoch+1}, Time:{end_epoch-start_epoch}, MAPE:{minPError}')

draw_curve(x_epoch, y_loss, y_err)