import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, TensorDataset

def to_tensor1(CNN_input_train,CNN_output_train,CNN_input_test,CNN_output_test,CNN_input_val,CNN_output_val):
  CNN_output_train = torch.tensor(CNN_output_train).float().to(device)
  CNN_input_train = torch.tensor(CNN_input_train).float().to(device)
  CNN_input_test = torch.tensor(CNN_input_test).float().to(device)
  CNN_output_test = torch.tensor(CNN_output_test).float().to(device)
  CNN_input_val = torch.tensor(CNN_input_val).float().to(device)
  CNN_output_val = torch.tensor(CNN_output_val).float().to(device)
  train_dataset = TensorDataset(CNN_input_train, CNN_output_train)
  train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
  validation_data = TensorDataset(CNN_input_val, CNN_output_val)
  validation_loader = DataLoader(validation_data, batch_size=50, shuffle=False)
  test_data = TensorDataset(CNN_input_test, CNN_output_test)
  test_data_loader = DataLoader(test_data, batch_size=50, shuffle=False)

  return train_dataset,train_loader,validation_data,validation_loader,test_data,test_data_loader