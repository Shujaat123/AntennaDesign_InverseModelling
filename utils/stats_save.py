import torch
import numpy as np
def save_stat(test_dec,test_inputs,test_en,test_targets,name,ave,save_path):
  for i in range(10):
    np.save(save_path+"/dLSE_test_dec1_"+name+"_"+str(ave),test_dec[i, :].squeeze(0).detach().cpu().numpy())
    np.save(save_path+"/dLSE_test_inputs1_"+name+"_"+str(ave),test_inputs[i, :].squeeze(0).detach().cpu().numpy())
    np.save(save_path+"/dLSE_test_inputs1_"+name+"_"+str(ave),test_inputs[i, :].squeeze(0).detach().cpu().numpy())
    np.save(save_path+"/dLSE_test_targets1_"+name+"_"+str(ave),test_targets[i, :].squeeze(0).detach().cpu().numpy())

    # np.save(f"/content/drive/My Drive/data for ML/resultsI/dLSE_test_dec1_{name}_{ave}",test_dec[i, :].squeeze(0).detach().cpu().numpy())
    # np.save(f"/content/drive/My Drive/data for ML/resultsI/dLSE_test_inputs1_{name}_{ave}",test_inputs[i, :].squeeze(0).detach().cpu().numpy())
    # np.save(f"/content/drive/My Drive/data for ML/resultsI/dLSE_test_inputs1_{name}_{ave}",test_inputs[i, :].squeeze(0).detach().cpu().numpy())
    # np.save(f"/content/drive/My Drive/data for ML/resultsI/dLSE_test_targets1_{name}_{ave}",test_targets[i, :].squeeze(0).detach().cpu().numpy())

  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_dec3_{ave}",test_dec[3, :].squeeze(0).detach().cpu().numpy())
  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_inputs3_{ave}",test_inputs[3, :].squeeze(0).detach().cpu().numpy())
  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_inputs3_{ave}",test_inputs[3, :].squeeze(0).detach().cpu().numpy())
  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_targets3_{ave}",test_targets[3, :].squeeze(0).detach().cpu().numpy())

  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_dec6_{ave}",test_dec[6, :].squeeze(0).detach().cpu().numpy())
  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_inputs6_{ave}",test_inputs[6, :].squeeze(0).detach().cpu().numpy())
  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_inputs6_{ave}",test_inputs[6, :].squeeze(0).detach().cpu().numpy())
  # np.save(f"/content/drive/My Drive/data for ML/results/dLSE_test_targets6_{ave}",test_targets[6, :].squeeze(0).detach().cpu().numpy())