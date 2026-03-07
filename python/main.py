# main.py
"""
Main pipeline for underwater image feature extraction

Extracts quality, texture/clarity, color/lighting, and deep features from images,
saves CSVs, generates distribution plots and correlation matrices, and combines
engineered features with PCA
"""

import os
import numpy as np
import pandas as pd
import torch

from io_utils import load_input_files
from feature_extraction import (
    process_quality_features,
    process_texture_clarity_features,
    process_color_and_lighting_features,
    process_deep_features,
    )
import config
from visualization import plot_distributions, plot_correlation_matrix, plot_pca_analysis

np.random.seed(config.random_seed)

# Helpers

def build_df(data, feature_names, filenames):
    """Combine filenames and feature array into a DataFrame"""
    df = pd.DataFrame(
        np.hstack([np.array(filenames).reshape(-1, 1), data]),
        columns=["filename"] + feature_names
        )
    df[feature_names] = df[feature_names].astype(float)
    return df


def align_on_filenames(*dfs):
    """Return copies of all dfs restricted to filenames present in all of them"""
    common = set(dfs[0]["filename"])
    for df in dfs[1:]:
        common &= set(df["filename"])

    n_dropped = max(len(df) for df in dfs) - len(common)
    if n_dropped > 0:
        print(f"Warning: {n_dropped} image(s) dropped due to extraction failures")

    return [df[df["filename"].isin(common)].reset_index(drop=True) for df in dfs]


if __name__ == "__main__":
    # Setup
    
    input_dir = config.input_dir
    out_dir = config.out_dir
    max_hw = config.max_hw
    pyiqa_metrics = config.pyiqa_quality_metrics
    n_pca_components = config.n_pca_components
    num_workers = config.num_workers

    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
    
    output_dir = os.path.join(out_dir, str(max_hw))
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature names
    
    quality_feature_names = ["ciqi", "uciqe", "uiqm"] + pyiqa_metrics
    
    texture_clarity_feature_names = (
        [f"laplacian_var_{c}" for c in ["h", "s", "v"]] +
        [f"sobel_var_{c}"     for c in ["h", "s", "v"]] +
        [f"lbp_hist_{c}_{i}"  for c in ["h", "s", "v"] for i in range(10)] +
        ["entropy"]
        )

    color_and_lighting_feature_names = (
        [f"color_hist_hsv_{c}_{i}"   for c in ["h", "s", "v"]   for i in range(4)] +
        [f"color_hist_ycrcb_{c}_{i}" for c in ["y", "cr", "cb"] for i in range(4)] +
        [f"color_hist_lab_{c}_{i}"   for c in ["l", "a", "b"]   for i in range(4)] +
        ["color_moments_L_mean",  "color_moments_L_std",  "color_moments_L_skewness",
         "color_moments_a_mean",  "color_moments_a_std",  "color_moments_a_skewness",
         "color_moments_b_mean",  "color_moments_b_std",  "color_moments_b_skewness",
         "mean_transmission", "std_transmission"]
        )
    
    # Load images
    
    input_files, _ = load_input_files(input_dir)
    print(f"Found {len(input_files)} images in '{input_dir}'.")

    # Extract features
    print("\nExtracting features...")

    quality_feats, quality_files = process_quality_features(input_files)
    quality_df = build_df(quality_feats, quality_feature_names, quality_files)
    quality_df.to_csv(os.path.join(output_dir, "quality_features.csv"), index=False)
    print(f"Quality features:           {quality_df.shape}")

    texture_feats, texture_files = process_texture_clarity_features(input_files, num_workers=num_workers)
    texture_clarity_df = build_df(texture_feats, texture_clarity_feature_names, texture_files)
    texture_clarity_df.to_csv(os.path.join(output_dir, "texture_clarity_features.csv"), index=False)
    print(f"Texture/clarity features:   {texture_clarity_df.shape}")

    color_feats, color_files = process_color_and_lighting_features(input_files, num_workers=num_workers)
    color_lighting_df = build_df(color_feats, color_and_lighting_feature_names, color_files)
    color_lighting_df.to_csv(os.path.join(output_dir, "color_and_lighting_features.csv"), index=False)
    print(f"Color/lighting features:    {color_lighting_df.shape}")

    deep_feats, deep_files = process_deep_features(input_files)
    deep_feature_names = [f"deep_feature_{i}" for i in range(deep_feats.shape[1])]
    deep_df = build_df(deep_feats, deep_feature_names, deep_files)
    deep_df.to_csv(os.path.join(output_dir, "deep_features.csv"), index=False)
    print(f"Deep features:              {deep_df.shape}")
    
    print("Feature extraction done")
    
    # Plots
    
    print("\nGenerating plots...")

    quality_df          = pd.read_csv(os.path.join(output_dir, "quality_features.csv"))
    texture_clarity_df  = pd.read_csv(os.path.join(output_dir, "texture_clarity_features.csv"))
    color_lighting_df   = pd.read_csv(os.path.join(output_dir, "color_and_lighting_features.csv"))
    deep_df             = pd.read_csv(os.path.join(output_dir, "deep_features.csv"))    
    
    feature_sets = {
        "quality_df":               quality_df,
        "texture_clarity_df":       texture_clarity_df,
        "color_and_lighting_df":    color_lighting_df,
        }
    
    for name, df in feature_sets.items():
        folder = os.path.join(output_dir, name)
        plot_distributions(df, folder, desc=name)
        plot_correlation_matrix(df, folder)
    
    # Deep features: PCA first, then visualize
    plot_pca_analysis(deep_df, os.path.join(output_dir, "deep_df"),
                      prefix="Deep", n_components=n_pca_components)
    
    # PCA
    
    print("\nCombining engineered features...")
    
    quality_df_a, texture_df_a, color_df_a = align_on_filenames(
        quality_df, texture_clarity_df, color_lighting_df
        )

    filenames_col = quality_df_a[["filename"]].reset_index(drop=True)
    combined_df = pd.concat(
        [filenames_col] +
        [df.drop(columns=["filename"]).reset_index(drop=True)
         for df in [quality_df_a, texture_df_a, color_df_a]],
        axis=1
        )
    combined_df.to_csv(os.path.join(output_dir, "combined_engineered_features.csv"), index=False)
    print(f"Combined engineered features: {combined_df.shape}")

    plot_pca_analysis(combined_df, os.path.join(output_dir, "combined_engineered"), 
                      prefix="Engineered", n_components=n_pca_components)
    
    print("\nPipeline complete. All outputs saved to:", output_dir)
    