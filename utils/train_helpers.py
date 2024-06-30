
import torch

def checkpoint(model,optimizer, filename):
  torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
  }, filename)


def resume(model,optimizer, filename):
  checkpoint = torch.load(filename)
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])