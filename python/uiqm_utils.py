# uiqm_utils.py
"""
Utility functions for computing the Underwater Image Quality Measure (UIQM)

Adapted from, with some changes
  https://github.com/xahidbuffon/FUnIE-GAN/blob/master/Evaluation/uqim_utils.py
"""
from scipy import ndimage
import numpy as np
import math


# Helpers

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """Calculates the asymmetric alpha-trimmed mean"""
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    trimmed_count = K - T_a_L - T_a_R
    if trimmed_count <= 0:
        return 0.0
    weight = 1.0 / trimmed_count
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    return weight * sum(x[s:e])


def s_a(x, mu):
    """Compute the variance of a 1-D array relative to mu"""
    val = sum(math.pow(pixel - mu, 2) for pixel in x)
    return val / len(x)


def _uicm(x):
    """Underwater Image Colorfulness Measure"""
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt(math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)

def _sobel(x):
    """Apply Sobel edge detection and normalise to [0, 255]"""
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    max_mag = np.max(mag)
    if max_mag > 0:
        mag *= 255.0 / max_mag  
    else:
        mag = np.zeros_like(mag) 
    return mag

def _eme(x, window_size):
    """Enhancement Measure Estimation over non-overlapping blocks"""
    k1 = x.shape[1] // window_size     # number of blocks along width
    k2 = x.shape[0] // window_size     # number of blocks along height
    w = 2.0 / (k1 * k2)
    x = x[:window_size * k2, :window_size * k1]     # trim image to exact multiple of blocks
    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)

            if min_ == 0.0 or max_ == 0.0:
                continue
            else:
                val += math.log(max_ / min_)
    return w * val

def _uism(x):
    """Underwater Image Sharpness Measure"""
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]

    Rs = _sobel(R)
    Gs = _sobel(G)
    Bs = _sobel(B)

    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    r_eme = _eme(R_edge_map, 8)
    g_eme = _eme(G_edge_map, 8)
    b_eme = _eme(B_edge_map, 8)

    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)

def _uiconm(x, window_size):
    """
    Underwater image contrast measure
        adapted from 
        https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
    """
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    w = -1.0 / (k1 * k2)
    x = x[:window_size * k2, :window_size * k1]
    alpha = 1  #entropy scale - higher helps with randomness
    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                continue
            else:
                val += alpha * math.pow(top / bot, alpha) * math.log(top / bot)
    return w * val


# PLIP operators (unused, kept for reference)

def plip_g(x, mu=1026.0):
    return mu - x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / gamma)

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta))

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))


def getUIQM(x):
    """
    Computes UIQM
        Coefficients from:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    """
    x = x.astype(np.float32)
    # c1 = 0.4680
    # c2 = 0.2745
    # c3 = 0.2576
    
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm