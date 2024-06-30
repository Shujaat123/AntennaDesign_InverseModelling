import torch
from utils.train_helpers import checkpoint, resume

def check_early_stopping(val_loss, best_validation_loss, best_validation_epoch, epoch, model, optimizer, early_stop_thresh,name,ave,save_path):
    current_val_loss = torch.mean(torch.tensor(val_loss))
    if current_val_loss < best_validation_loss:
        best_validation_loss = current_val_loss
        best_validation_epoch = epoch
        # checkpoint(model, optimizer, f"/content/drive/My Drive/data for ML/resultsI/{name}_{ave}.pth")
        checkpoint(model, optimizer, save_path+"/"+name+"_"+str(ave)+".pth")
    elif epoch - best_validation_epoch > early_stop_thresh:
        print(f"Early stopped training at epoch {epoch}. \nThe epoch of the best validation accuracy was {best_validation_epoch} with validation accuracy of {best_validation_loss}")
        return best_validation_loss, best_validation_epoch, True
    
    return best_validation_loss, best_validation_epoch, False
