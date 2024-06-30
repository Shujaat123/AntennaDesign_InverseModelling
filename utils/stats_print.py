import torch


def show_stats(test_loss,rmse_test,r_squared_test,test_ssim,test_psnr):
  print(f'\nTest Loss: {torch.mean(torch.tensor(test_loss)):.4f}',
        f'RMSE Test: {torch.mean(torch.tensor(rmse_test)):.4f}',
        f'R-squared Test: {torch.mean(torch.tensor(r_squared_test)):.4f}',
        f'Test SSIM: {torch.mean(torch.tensor(test_ssim)):.4f}',
        f'Test PSNR: {torch.mean(torch.tensor(test_psnr)):.4f}',
        )