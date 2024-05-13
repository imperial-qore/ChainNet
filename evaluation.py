import torch
from ChainNet import Net
from input import ChainNetDataset, ProcessData, CollateBatch
from torch.utils.data import DataLoader

@torch.no_grad()
def test(model, test_loader):
    model.eval()

    for data in test_loader:
        prediction = model(data)
    return prediction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainedmodel = torch.load('model.pth')
model = Net(num_iterations=8, size_realnode=64, size_hypernode=64, num_readoutneurons=64, num_heads=2, negative_slope=0.2, dropout=0.0)
model.load_state_dict(trainedmodel['model_state_dict'])
model.to(device)

data_list = ProcessData(root='Data/test_TypeII/', numSamples=10000)
test_dataset = ChainNetDataset(data_list)
test_loader = DataLoader(test_dataset, batch_size=10000, collate_fn=CollateBatch)
prediction = test(model, test_loader)