# feature_extraction.py
"""
Feature extraction functions: quality metrics, texture/clarity, color/lighting, deep features
"""

import os
import numpy as np
import cv2
import scipy.stats
import torch
from PIL import Image
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from concurrent.futures import ProcessPoolExecutor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.nn import Sequential
from pyiqa import create_metric
from pyiqa.utils.img_util import imread2tensor

import config
from io_utils import read_and_resize_image
from uiqm_utils import getUIQM

# Setup

pyiqa_quality_metrics = config.pyiqa_quality_metrics

device = torch.device("cpu")
# NOTE: _pyiqa_metrics and _resnet_model are process-local
# Do NOT use ProcessPoolExecutor for quality or deep feature extraction
_pyiqa_metrics = None
_resnet_model = None
_resnet_transform = None


# Quality features

def pyiqa_metrics(image, metric_names):
    """Compute IQA scores using the pyiqa library"""
    global _pyiqa_metrics
    if _pyiqa_metrics is None:
        _pyiqa_metrics = {name: create_metric(name, device=device).eval() for name in metric_names}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    img_tensor = imread2tensor(img, rgb=True).to(device).unsqueeze(0)

    results = {}

    with torch.no_grad():
        for name in metric_names:
            metric = _pyiqa_metrics[name]
            score = metric(img_tensor).item()
            pref = "lower is better" if metric.lower_better else "higher is better"
            results[name] = (score, pref)

    return results


def ciqi(image, l_contrast_weight=0.0, l_std_weight=0.5141, chroma_std_weight=0.4859):
    """
    Compute the CIQI (Color Image Quality Index) with coefficients optimized for underwater images
    Reference: "Color image quality measures and retrieval, Fu, 2006"
    coefficients from "An Underwater Color Image Quality Evaluation Metric, Yang et al., 2015"
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    l_channel = l_channel.astype(np.float32)
    a_channel = a_channel.astype(np.float32)
    b_channel = b_channel.astype(np.float32)

    l_contrast = np.percentile(l_channel, 99) - np.percentile(l_channel, 1)
    l_std = np.std(l_channel)

    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)
    chroma_std = np.std(chroma)
    ciqi_score = l_contrast_weight * l_contrast + l_std_weight * l_std + chroma_std_weight * chroma_std

    return ciqi_score


def uciqe(image, sigma_c_weight=0.4680, con_l_weight=0.2745, mu_s_weight=0.2576):
    """
    Compute the Underwater Color Image Quality Evaluation (UCIQE) metric
    Reference: "An Underwater Color Image Quality Evaluation Metric, Yang et al., 2015"
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    l_channel = l_channel.astype(np.float32)
    a_channel = a_channel.astype(np.float32)
    b_channel = b_channel.astype(np.float32)

    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2)

    sigma_c = np.std(chroma)
    con_l = np.percentile(l_channel, 99) - np.percentile(l_channel, 1)

    valid_pixels = (chroma > 0) & (l_channel > 0)

    saturation = np.zeros_like(chroma)
    saturation[valid_pixels] = chroma[valid_pixels] / l_channel[valid_pixels]

    mu_s = np.mean(saturation)

    uciqe_score = sigma_c_weight * sigma_c + con_l_weight * con_l + mu_s_weight * mu_s

    return uciqe_score


def p_quality(image):
    """Return a 1-D array of all quality features for a single image"""
    uciqe_score = uciqe(image)
    ciqi_score = ciqi(image)
    uiqm_score = getUIQM(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    pyiqa_scores = pyiqa_metrics(image, pyiqa_quality_metrics)
    pyiqa_values = [score[0] for score in pyiqa_scores.values()]

    return np.hstack([ciqi_score, uciqe_score, uiqm_score, pyiqa_values]).ravel()


def process_quality_features(input_files):
    """Extract quality features for all input files"""
    feats, valid_files = [], []
    for p in tqdm(input_files, desc="Quality Assessment Metrics"):
        try:
            image = read_and_resize_image(p)
            feats.append(p_quality(image))
            valid_files.append(p)
        except Exception as e:
            print(f"\nWarning: skipping {p} in quality features - {e}")
    return np.array(feats), valid_files


# Texture and clarity 

def p_blur(image, sobel_kernel=5, blur_kernel=(5, 5)):
    """Compute Laplacian and Sobel variance per HSV channel"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    laplacian_variance_hsv = []
    sobel_variance_hsv = []

    for channel in [h, s, v]:
        blurred = cv2.GaussianBlur(channel, blur_kernel, 0)
        laplacian_variance = np.var(cv2.Laplacian(blurred, cv2.CV_64F))
        laplacian_variance_hsv.append(laplacian_variance)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel_variance = np.var(np.hypot(sobel_x, sobel_y))
        sobel_variance_hsv.append(sobel_variance)
    return np.hstack([laplacian_variance_hsv, sobel_variance_hsv]).ravel()


def p_content_lbp(image, P=8, blur_kernel=(5, 5)):
    """Compute LBP histogram per HSV channel"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    lbp_hist_hsv = []
    for channel in [h, s, v]:
        blurred = cv2.GaussianBlur(channel, blur_kernel, 0)
        lbp = local_binary_pattern(blurred, P, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), density=True)
        lbp_hist_hsv.append(lbp_hist)

    return np.hstack(lbp_hist_hsv).ravel()


def compute_entropy(image):
    """Compute Shannon entropy of the grayscale histograms"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist_normalized = hist / np.sum(hist)

    mask = hist_normalized > 0
    entropy = -np.sum(hist_normalized[mask] * np.log2(hist_normalized[mask]))

    return entropy


def _compute_texture_clarity_from_path(file_path):
    """Returns (file_path, feature_vector) or (file_path, None) on failure"""
    try:
        image = read_and_resize_image(file_path)
        blur_features = p_blur(image)
        content_lbp = p_content_lbp(image)
        entropy = compute_entropy(image)
        return file_path, np.hstack([blur_features, content_lbp, entropy])
    except Exception as e:
        print(f"\nWarning: skipping {file_path} in texture/clarity worker - {e}")
        return file_path, None


def process_texture_clarity_features(input_files, num_workers=None):
    """Extract texture and clarity features for all input files"""
    if num_workers is None:
        num_workers = min(max(1, (os.cpu_count() or 1) - 1), len(input_files))

    chunksize = max(1, len(input_files) // (num_workers * 4))

    feats, valid_files = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for file_path, result in tqdm(
                ex.map(_compute_texture_clarity_from_path, input_files, chunksize=chunksize),
                total=len(input_files), desc="Texture and Clarity Analysis"
                ):
            if result is not None:
                feats.append(result)
                valid_files.append(file_path)

    return np.array(feats), valid_files


# Color and lighting

def p_content_color_histogram(image, color_space, num_bins=4):
    """
    Compute a color histogram for the given color space

    Uses correct per-channel ranges:
    - HSV: H in [0, 180] (OpenCV convention), S and V in [0, 256]
    - YCrCb: all channels in [0, 256]
    - Lab: all channels in [0, 256] (OpenCV 8-bit scaled)
    """
    hist_bin_edges = {
    'hsv':   [np.linspace(0, 180, num_bins+1), np.linspace(0, 256, num_bins+1), np.linspace(0, 256, num_bins+1)],
    'ycrcb': [np.linspace(0, 256, num_bins+1), np.linspace(0, 256, num_bins+1), np.linspace(0, 256, num_bins+1)],
    'lab':   [np.linspace(0, 256, num_bins+1), np.linspace(0, 256, num_bins+1), np.linspace(0, 256, num_bins+1)],
    }
    if color_space.lower() == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space.lower() == 'ycrcb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space.lower() == 'lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    channels = cv2.split(image)
    bin_edges = hist_bin_edges[color_space.lower()]
    hist_features = []
    for channel, edges in zip(channels, bin_edges):
        hist, _ = np.histogram(channel.ravel(), bins=edges, density=True)
        hist_features.extend(hist)

    return np.array(hist_features)


def p_content_color_moments(image):
    """Compute mean, std, and skewness per channel"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    mean = np.mean(lab_image, axis=(0, 1))
    std_dev = np.std(lab_image, axis=(0, 1))
    skewness = scipy.stats.skew(lab_image.reshape(-1, 3), axis=0)

    return np.hstack([mean, std_dev, skewness])


def compute_dark_channel(image, patch_size=15):
    """
    Compute the dark channel of the image
    Reference: "Single Image Haze Removal Using Dark Channel Prior, He et al., 2009"
    """
    image = image.astype(np.float32)
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel


def compute_transmission(dark_channel, omega=0.95):
    """Estimate transmission map via the dark channel prior"""
    transmission = 1 - omega * dark_channel / 255.0

    return np.clip(transmission, 0, 1)


def _compute_color_lighting_from_path(file_path, color_spaces):
    """Returns (file_path, feature_vector) or (file_path, None) on failure"""
    try:
        image = read_and_resize_image(file_path)
        color_histograms = np.hstack([p_content_color_histogram(image, cs) for cs in color_spaces])
        color_moments = p_content_color_moments(image)
        dark_channel = compute_dark_channel(image)
        transmission = compute_transmission(dark_channel)
        return file_path, np.hstack([color_histograms, color_moments, np.mean(transmission), np.std(transmission)])
    except Exception as e:
        print(f"\nWarning: skipping {file_path} in color/lighting worker - {e}")
        return file_path, None


def _compute_color_lighting_from_args(args):
    file_path, color_spaces = args

    return _compute_color_lighting_from_path(file_path, color_spaces)


def process_color_and_lighting_features(input_files, color_spaces=None, num_workers=None):
    """Extract color and lighting features for all input files"""
    if color_spaces is None:
        color_spaces = ['hsv', 'ycrcb', 'lab']
    if num_workers is None:
        num_workers = min(max(1, (os.cpu_count() or 1) - 1), len(input_files))

    chunksize = max(1, len(input_files) // (num_workers * 4))
    args = [(p, tuple(color_spaces)) for p in input_files]

    feats, valid_files = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for file_path, result in tqdm(
                ex.map(_compute_color_lighting_from_args, args, chunksize=chunksize), 
                total=len(input_files), desc="Color and Lighting Analysis"
                ):
            if result is not None:
                feats.append(result)
                valid_files.append(file_path)

    return np.array(feats), valid_files


# Deep features (ResNet50)

def _get_resnet():
    global _resnet_model, _resnet_transform
    if _resnet_model is None:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        _resnet_model = Sequential(*list(model.children())[:-1]).to(device).eval()
        _resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    return _resnet_model, _resnet_transform


def extract_deep_features_resnet50(image):
    """Extract a 2048-D feature vector from pretrained ResNet50 (no FC layer)"""
    model, transform = _get_resnet()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = transform(pil_image).to(device).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)  # Output: (batch, 2048, 1, 1)

    return features.cpu().numpy().flatten()


def process_deep_features(input_files):
    """Extract deep features for all input files"""
    feats, valid_files = [], []
    for p in tqdm(input_files, desc="Extracting Deep Features"):
        try:
            image = read_and_resize_image(p)
            feats.append(extract_deep_features_resnet50(image))
            valid_files.append(p)
        except Exception as e:
            print(f"\nWarning: skipping {p} in deep features - {e}")

    return np.array(feats), valid_files


if __name__ == "__main__":
    # Quick check of all feature extraction functions on a small sample

    import time

    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
    SAMPLES_DIR = config.input_dir # change as needed. eg., "samples"
    MAX_TEST_IMAGES = 10  # change as needed

    if not os.path.isdir(SAMPLES_DIR):
        raise FileNotFoundError(f"Samples folder not found: '{SAMPLES_DIR}'")

    sample_files = sorted(os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR)
        if os.path.splitext(f.lower())[1] in config.img_extensions)

    if len(sample_files) == 0:
        raise FileNotFoundError(f"No image files found in '{SAMPLES_DIR}'")

    test_files = sample_files[:min(MAX_TEST_IMAGES, len(sample_files))]
    print(f"Found {len(sample_files)} images. Testing on {len(test_files)} images from '{SAMPLES_DIR}/'.")

    # Single-image checks
    img_path = test_files[0]
    img = read_and_resize_image(img_path)

    print(f"\nSingle image: {os.path.basename(img_path)}  shape={img.shape}  dtype={img.dtype}")

    t0 = time.perf_counter()
    ciqi_val = ciqi(img)
    uciqe_val = uciqe(img)
    print(f"ciqi={ciqi_val:.6f}  uciqe={uciqe_val:.6f}")

    pyiqa_results = pyiqa_metrics(img, pyiqa_quality_metrics)
    print("pyiqa metrics:")
    for k, (score, pref) in pyiqa_results.items():
        print(f"  - {k}: {score:.6f} ({pref})")

    q = p_quality(img)
    print(f"p_quality vector length={q.shape[0]}  first5={q[:5]}")

    blur = p_blur(img)
    lbp = p_content_lbp(img)
    ent = compute_entropy(img)
    print(f"p_blur length={blur.shape[0]}  p_content_lbp length={lbp.shape[0]}  entropy={ent:.6f}")

    for cs in ["hsv", "ycrcb", "lab"]:
        hist = p_content_color_histogram(img, cs, num_bins=4)
        print(f"color_hist {cs} length={hist.shape[0]}")

    moments = p_content_color_moments(img)
    dc = compute_dark_channel(img)
    tr = compute_transmission(dc)
    print(f"color_moments length={moments.shape[0]}  dark_channel shape={dc.shape}  transmission mean={np.mean(tr):.6f}")

    deep = extract_deep_features_resnet50(img)
    print(f"deep features length={deep.shape[0]}")

    print(f"Single-image checks done in {time.perf_counter() - t0:.2f}s\n")

    # Batch checks (process_* functions)
    print("\nBatch extraction (process_*):")

    t0 = time.perf_counter()

    Q, q_files = process_quality_features(test_files)
    print(f"process_quality_features: shape={Q.shape}  valid={len(q_files)}/{len(test_files)}\n")

    T, t_files = process_texture_clarity_features(test_files)
    print(f"process_texture_clarity_features: shape={T.shape}  valid={len(t_files)}/{len(test_files)}\n")

    C, c_files = process_color_and_lighting_features(test_files)
    print(f"process_color_and_lighting_features: shape={C.shape}  valid={len(c_files)}/{len(test_files)}\n")

    D, d_files = process_deep_features(test_files)
    print(f"process_deep_features: shape={D.shape}  valid={len(d_files)}/{len(test_files)}\n")

    print(f"\nAll batch checks done in {time.perf_counter() - t0:.2f}s")
