# visualization.py
"""
Visualization utilities: feature distribution histograms and correlation heatmaps
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA


def _clean_labels(matrix):
    """Replace underscores with spaces in DataFrame index and column labels"""
    matrix.columns = matrix.columns.str.replace("_", " ")
    matrix.index = matrix.index.str.replace("_", " ")
    return matrix


def _to_numeric(df, drop_col="filename"):
    """Return a copy of df with all non-filename columns cast to float"""
    numeric_cols = [c for c in df.columns if c != drop_col]
    out = df.copy()
    out[numeric_cols] = out[numeric_cols].astype(float)
    return out


def plot_distributions(df, save_folder, desc="", drop_col="filename"):
    """Save a histogram PNG for each feature column in df (skips drop_col)"""
    os.makedirs(save_folder, exist_ok=True)
    df = _to_numeric(df, drop_col)

    for col in tqdm(df.columns[1:], desc=f"Histograms {desc}"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[col].dropna(), bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(os.path.join(save_folder, f"{col}_distribution.png"), dpi=300)
        plt.close(fig)


def plot_correlation_matrix(df, save_folder, drop_col="filename"):
    """Save annotated and unannotated correlation heatmaps side by side"""
    os.makedirs(save_folder, exist_ok=True)
    df = _to_numeric(df, drop_col)

    corr = _clean_labels(df.drop(columns=[drop_col]).corr())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, annotate, title in zip(axes, [True, False], ["Annotated", "Unannotated"]):
        sns.heatmap(corr, annot=annotate, cmap="coolwarm", fmt=".2f" if annotate else "",
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation Matrix ({title})", fontsize=14)
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
        
    plt.tight_layout()
    fig.savefig(os.path.join(save_folder, "correlation_comparison.png"), dpi=300)
    plt.close(fig)


def plot_pca_analysis(df, save_folder, prefix, n_components=10, drop_col="filename"):
    """Apply PCA to df, then save histograms and correlation heatmap for the components"""
    os.makedirs(save_folder, exist_ok=True)
    df = _to_numeric(df, drop_col)
    data = df.drop(columns=[drop_col])

    max_components = min(n_components, data.shape[0], data.shape[1])
    if max_components < n_components:
        print(f"Warning: reducing n_components {n_components} -> {max_components} "
              f"for {prefix} (shape={data.shape})")
        n_components = max_components
        
    total_null = data.isnull().sum().sum()
    if total_null > 0:
        print(f"Total null values in {prefix}: {total_null}")
        print(f"Null values per column:\n{data.isnull().sum()[data.isnull().sum() > 0]}")
        print("Filling NaN values with column means")
        data = data.fillna(data.mean())
    else:
        print(f"No null values detected in {prefix}")

    components = PCA(n_components=n_components).fit_transform(data)
    pca_df = pd.DataFrame(components, columns=[f"{prefix}_PCA_{i+1}" for i in range(n_components)])

    filenames = df[[drop_col]].reset_index(drop=True)
    pca_with_names = pd.concat([filenames, pca_df], axis=1)
    
    plot_distributions(pca_with_names, save_folder, desc=f"{prefix} PCA")
    plot_correlation_matrix(pca_with_names, save_folder)
    
    return pca_df
