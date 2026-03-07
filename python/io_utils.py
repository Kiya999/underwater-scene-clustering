# io_utils.py
"""
Utility functions for loading and preprocessing input images
"""

import os
import csv
import cv2
from tqdm import tqdm

import config

# Setup

max_hw = config.max_hw
out_dir = config.out_dir
img_extensions = config.img_extensions


def read_and_resize_image(input_path):
    """Read an image and resize so the longer side does not exceed max_hw"""
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed - file not found or unreadable: {input_path}")
    
    height, width = img.shape[:2]

    if max_hw != "na":
        aspect_ratio = width / height
        if width >= height:
            new_width = max_hw
            new_height = int(max_hw / aspect_ratio)
        else:
            new_height = max_hw
            new_width = int(max_hw * aspect_ratio)

        img = cv2.resize(img, (new_width, new_height))

        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "resize_log.csv"), "a", newline="") as f:
                csv.writer(f).writerow([input_path, height, width, new_width, new_height])
        except Exception as e:
            print(f"Warning: could not write resize_log.csv - {e}")

    return img


def load_input_files(input_dir):
    """Return sorted lists of full file paths and filenames from input_dir"""
    files_path, files_name = [], []
    for f in tqdm(sorted(os.listdir(input_dir)), desc="Scanning input files"):
        if os.path.splitext(f.lower())[1] in img_extensions:
            files_path.append(os.path.join(input_dir, f))
            files_name.append(f)
    return files_path, files_name
