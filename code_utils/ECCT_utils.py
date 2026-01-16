import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def SNR_to_std(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()