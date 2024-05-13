import torch
import matplotlib.pyplot as plt

def draw_curve(x_epoch, y_loss, y_err):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    ax.plot(x_epoch, y_loss['train'], label = "train")
    ax.plot(x_epoch, y_loss['val'], label = "val")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    plt.grid(True)
    fig.savefig('./loss_curve.pdf')

    fig, ax = plt.subplots()
    ax.plot(x_epoch, y_err['val'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('MAPE')
    fig.savefig('./MAPE_curve.pdf')

class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, criterion, optimizer 
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, './model.pth')
        return self.best_valid_loss