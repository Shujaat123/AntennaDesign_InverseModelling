import torch
from sklearn.metrics import mean_squared_error, r2_score

import torch
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim


def evaluate_statI(inn, targets, inputs, decoded_input, criterion, loss_fn21, loss_fn2, lambda_w1, lambda_w2, 
                   lambda_w3=None, outp=None):
    train_loss = []
    train_psnr = []
    rmse_train = []
    r_squared_train = []
    train_ssim = []
    
    for ii in range(inn.shape[0]):
        if lambda_w3 is not None and outp is not None:
            loss = (lambda_w1 * criterion(inn[ii, :], targets[ii, :]) + 
                    lambda_w2 * loss_fn21(inputs[ii, :], decoded_input[ii, :]) +
                    lambda_w3 * criterion(outp[ii, :], inputs[ii, :]))
            psnr_value = 10 * torch.log10(1 / loss_fn2(inputs[ii, :], outp[ii, :]))
            ssim_value = ssim(
                inputs[ii, :].cpu().detach().numpy(),
                outp[ii, :].cpu().detach().numpy(),
                win_size=min(7, min(inputs[ii].shape)),
                gaussian_weights=False,
                use_sample_covariance=False
            )
        else:
            loss = (lambda_w1 * criterion(inn[ii, :], targets[ii, :]) + 
                    lambda_w2 * loss_fn21(inputs[ii, :], decoded_input[ii, :]))
            psnr_value = 10 * torch.log10(1 / loss_fn2(inputs[ii, :], decoded_input[ii, :]))
            ssim_value = ssim(
                inputs[ii, :].cpu().detach().numpy(),
                decoded_input[ii, :].cpu().detach().numpy(),
                win_size=min(7, min(inputs[ii].shape)),
                gaussian_weights=False,
                use_sample_covariance=False
            )
        
        train_loss.append(loss.item())
        train_psnr.append(psnr_value)
        rmse_train.append(mean_squared_error(inn[ii, :].cpu().numpy(), targets[ii, :].cpu().numpy()))
        r_squared_train.append(r2_score(inn[ii, :].cpu().numpy(), targets[ii, :].cpu().numpy()))
        train_ssim.append(ssim_value)

    return train_loss, train_psnr, rmse_train, r_squared_train, train_ssim

# Example usage:
# train_loss, train_psnr, rmse_train, r_squared_train, train_ssim = evaluate_statI(
#     inn, targets, inputs, decoded_input, criterion, loss_fn21, loss_fn2, lambda_w1, lambda_w2, lambda_w3, outp)



# def evaluate_statI(inn, targets, inputs, decoded_input, criterion, loss_fn21, loss_fn2, lambda_w1, lambda_w2,lambda_w3):
#     loss = []
#     psnr = []
#     train = []
#     rmse=[]
#     r_squared = []
#     ssimI = []

#     for ii in range(inn.shape[0]):
#         loss.append(
#             (lambda_w1) * criterion(inn[ii, :], targets[ii, :]) + (lambda_w2) * loss_fn21(inputs[ii, :], decoded_input[ii, :])
#         )
#         psnr.append(
#             10 * torch.log10(1 / loss_fn2(inputs[ii, :], decoded_input[ii, :]))
#         )
#         rmse.append(
#             mean_squared_error(inn[ii, :].cpu().numpy(), targets[ii, :].cpu().numpy())
#         )
#         r_squared.append(
#             r2_score(inn[ii, :].cpu().numpy(), targets[ii, :].cpu().numpy())
#         )
#         win_size = min(7, min(inputs[ii].shape))  # Ensure odd value
#         ssimI.append(
#             ssim(
#                 inputs[ii, :].cpu().detach().numpy(),
#                 decoded_input[ii, :].cpu().detach().numpy(),
#                 win_size=win_size,
#                 gaussian_weights=False,
#                 use_sample_covariance=False
#             )
#         )

#     return loss, psnr, rmse, r_squared, ssimI



