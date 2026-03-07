# config.py

import os

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(_base, "data", "input", "U45")
out_dir   = os.path.join(_base, "data", "output", "U45")

max_hw = "na" # set to "na" to skip resizing

num_workers = None  # defaults to os.cpu_count() - 1

pyiqa_quality_metrics = ["brisque", "uranker", "arniqa", "topiq_nr"]

img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

n_pca_components = 10

random_seed = 23
