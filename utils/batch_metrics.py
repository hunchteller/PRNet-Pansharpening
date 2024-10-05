import torch
import numpy as np
# from .q2n import Q2N
from einops import rearrange, repeat
import torch.nn.functional as F
import math

eps = 1e-10

def _ssim(img1, img2, bit):

    img1 = img1.float() / 2 ** bit
    img2 = img2.float() / 2 ** bit
    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = create_window(window_size, sigma, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean((1, 2, 3))


from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# Cross-correlation matrix
def batch_cross_correlation(H_fuse, H_ref):
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, -1, keepdim=True)
    mean_ref = torch.mean(H_ref_reshaped, -1, keepdim=True)

    CC = torch.sum((H_fuse_reshaped - mean_fuse) * (H_ref_reshaped - mean_ref), -1) / torch.sqrt(
        torch.sum((H_fuse_reshaped - mean_fuse) ** 2, -1) * torch.sum((H_ref_reshaped - mean_ref) ** 2, -1))

    CC = torch.nansum(CC, dim=-1) / N_spectral
    CC = CC
    return CC


# Spectral-Angle-Mapper (SAM)
def batch_SAM(H_fuse, H_ref):
    # Compute number of spectral bands
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)
    N_pixels = H_fuse_reshaped.size(-1)

    # Calculating inner product
    inner_prod = torch.sum(H_fuse_reshaped * H_ref_reshaped, 1)
    fuse_norm = torch.sum(H_fuse_reshaped ** 2, dim=1)
    ref_norm = torch.sum(H_ref_reshaped ** 2, dim=1)
    denom = (fuse_norm * ref_norm).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / denom), dim=1) / N_pixels)
    return SAM


# Root-Mean-Squared Error (RMSE)
def batch_RMSE(H_fuse, H_ref):
    # Rehsaping fused and reference data
    batch = H_fuse.size(0)
    H_fuse_reshaped = H_fuse.view(batch, -1)
    H_ref_reshaped = H_ref.view(batch, -1)

    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=-1) / H_fuse_reshaped.size(1))
    return RMSE.sum()


# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def batch_ERGAS(H_fuse, H_ref, beta):
    # Compute number of spectral bands
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)
    N_pixels = H_fuse_reshaped.size(-1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=-1) / N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=-1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100 * (1 / beta) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2, dim=-1) / N_spectral)
    return ERGAS


# Peak SNR (PSNR)
def batch_PSNR(H_fuse, H_ref, Bit):
    # Compute number of spectral bands
    batch, N_spectral = H_fuse.size(0), H_fuse.size(1)

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(batch, N_spectral, -1)
    H_ref_reshaped = H_ref.view(batch, N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=-1) / H_fuse_reshaped.size(-1))

    # Calculating max pixel
    Max_pixel = 2**Bit - 1

    # Calculating PSNR
    PSNR = torch.nansum(10 * torch.log10(torch.div(Max_pixel, rmse) ** 2), dim=1) / N_spectral

    return PSNR

def get_HC(x):
    return torch.cat((x[..., :1], -x[..., 1:]), -1)

def onion_mult(onion1, onion2):
    """
    onion1: 
    onion2:
    """
    N = onion1.size(-1)
    if N > 1:
        L = N//2
        a = onion1[..., :L]
        b = onion1[..., L:]
        b = get_HC(b)

        c = onion2[..., :L]
        d = onion2[..., L:]
        d = get_HC(d)

        if N == 2:
            ris = torch.cat([a*c - d*b, a*d + c*b], -1)
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, get_HC(b))
            ris3 = onion_mult(get_HC(a), d)
            ris4 = onion_mult(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat([aux1, aux2], -1)
    else:
        ris = onion1 * onion2
    return ris

def onion_mult2D(onion1, onion2):
    """
    
    """
    N3 = onion1.size(-1)
    if N3 > 1:
        L = N3//2
        a = onion1[..., :L]
        b = onion1[..., L:]
        b = get_HC(b)

        c = onion2[..., :L]
        d = onion2[..., L:]
        d = get_HC(d)

        if N3 == 2:
            ris = torch.cat([a*c - d*b, a*d + c*b], -1)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, get_HC(b))
            ris3 = onion_mult2D(get_HC(a), d)
            ris4 = onion_mult2D(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat([aux1, aux2], -1)
    else:
        ris = onion1 * onion2
    return ris


def norm_blocco(x, eps=1e-7):
    m = x.mean(dim=-2, keepdim=True)
    s = x.std(dim=-2, keepdim=True) 
    x = (x - m) / (s + eps) + 1
    return x, m, s+eps

def batch_Q2N(pr, gt, block_size=32):
    eps = 1e-7
    dat2 = pr.long().float()
    dat1 = gt.long().float()

    # dat2 = pr.float()
    # dat1 = gt.float()

    dat2 = rearrange(dat2, 'nb c (nh h) (nw w) -> nb (nh nw) (h w) c', h=block_size, w=block_size)
    dat1 = rearrange(dat1, 'nb c (nh h) (nw w) -> nb (nh nw) (h w) c', h=block_size, w=block_size)



    dat1, m, s = norm_blocco(dat1)
    # assert (m==0).any()
    mask = (m == 0).float()
    s = 1 * mask + s * (1-mask)
    dat2 = (dat2 - m) / s  + 1

    dat2 = get_HC(dat2)
    
    ratio = block_size * block_size 
    ratio = ratio / (ratio - 1)

    m1 = dat1.mean(-2)
    m2 = dat2.mean(-2)

    mod_q1m = m1.pow(2).sum(-1).sqrt()
    mod_q2m = m2.pow(2).sum(-1).sqrt()  

    mod_q1 = dat1.pow(2).sum(-1).sqrt()
    mod_q2 = dat2.pow(2).sum(-1).sqrt()

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m.pow(2) + mod_q2m.pow(2)

    int1 = ratio * mod_q1.pow(2).mean(-1)
    int2 = ratio * mod_q2.pow(2).mean(-1)
    term3 = int1 + int2 - ratio* term4
    term3 = term3.unsqueeze(-1)

    mean_bias = (2 * term2 / term4).unsqueeze(-1)
    
    

    mask = (term3 == 0).float()
    # q = torch.zeros_like(mean_bias)
  
    cbm = 2 / (term3+eps)
    qu = onion_mult2D(dat1, dat2)
    qm = onion_mult(m1, m2)
    qv = ratio * qu.mean(-2)
    q = qv - ratio * qm
    q = q * mean_bias * cbm
    q = q * (1-mask) + mask * mean_bias
    q = (q**2).sum(-1).sqrt().mean(-1)
    return q

def RR_Metrics(pr, gt, Bit):
    sam = batch_SAM(pr, gt).mean()
    ergas = batch_ERGAS(pr, gt, beta=4).mean()
    cc = batch_cross_correlation(pr, gt).mean()
    ssim = _ssim(pr, gt, Bit).mean()
    psnr = batch_PSNR(pr, gt, Bit).mean()
    q2n = batch_Q2N(pr, gt).mean()
    return {"SAM":sam, "ERGAS":ergas, "CC":cc, "SSIM":ssim, "PSNR":psnr, 'Q2N':q2n}



if __name__ == '__main__':
    x = torch.rand(1, 4, 128, 128) * 2**10
    y = torch.rand(1, 4, 128, 128) * 2**10
    x = x.long().float()
    y = y.long().double()

    print(RR_Metrics(x, x, 10))
