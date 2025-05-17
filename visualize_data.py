#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_visualization_and_checks.py
---------------------------------
Script to visualize and perform integrity checks on the generated
connectivity matrix tensors (.pt files) before CVAE training.

- Loads data and metadata.
- Checks for NaNs, Infs, and dimension consistency.
- Visualizes example matrices and value distributions.
- Reports class balance (AD, CN, Other).
"""

import argparse
import logging
from pathlib import Path
import random
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pt_file(file_path: Path) -> Tuple[torch.Tensor | None, Dict[str, Any] | None]:
    """Loads a .pt file containing data tensor and metadata."""
    try:
        content = torch.load(file_path, map_location=torch.device('cpu')) # Load to CPU
        if isinstance(content, dict) and "data" in content and "meta" in content:
            return content["data"], content["meta"]
        else:
            logger.warning(f"File {file_path} does not have the expected structure (dict with 'data' and 'meta' keys).")
            return None, None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None

def plot_connectivity_matrix(matrix: np.ndarray, title: str, output_path: Path | None = None, metric_name: str = ""):
    """Plots a single connectivity matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap="viridis", cbar=True, square=True) # 'viridis' is often good for FC
    full_title = f"{title}\nMetric: {metric_name} (Shape: {matrix.shape})"
    plt.title(full_title)
    plt.xlabel("ROI Index")
    plt.ylabel("ROI Index")
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved matrix plot to {output_path}")
        plt.close()
    else:
        plt.show()

def plot_value_distribution(values: np.ndarray, title: str, output_path: Path | None = None, metric_name: str = "", bins: int = 50):
    """Plots a histogram of matrix values (typically off-diagonal)."""
    if values.size == 0:
        logger.warning(f"No values to plot for histogram: {title} - {metric_name}")
        return
    plt.figure(figsize=(8, 6))
    sns.histplot(values, bins=bins, kde=True)
    full_title = f"Value Distribution for {title}\nMetric: {metric_name}"
    plt.title(full_title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved distribution plot to {output_path}")
        plt.close()
    else:
        plt.show()

def plot_class_distribution(group_counts: pd.Series, output_path: Path | None = None):
    """Plots the distribution of AD, CN, and Other groups."""
    plt.figure(figsize=(8, 6))
    group_counts.plot(kind='bar')
    plt.title("Class Distribution (AD vs CN vs Other)")
    plt.xlabel("Group")
    plt.ylabel("Number of Subjects")
    plt.xticks(rotation=0)
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved class distribution plot to {output_path}")
        plt.close()
    else:
        plt.show()


def main(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        logger.error(f"Input directory {input_dir} does not exist or is not a directory.")
        return

    pt_files = sorted(list(input_dir.glob("*.pt")))
    if not pt_files:
        logger.error(f"No .pt files found in {input_dir}.")
        return

    logger.info(f"Found {len(pt_files)} .pt files in {input_dir}.")

    all_metadata = []
    example_data_to_plot = [] # Store tuples of (tensor, meta) for a few subjects
    
    # --- Data Loading and Initial Properties ---
    first_data, first_meta = load_pt_file(pt_files[0])
    if first_data is None or first_meta is None:
        logger.error("Could not load the first .pt file to get essential properties. Exiting.")
        return

    expected_num_metrics = first_data.shape[0]
    expected_num_rois = first_data.shape[1] # Assuming square matrices, shape[1] == shape[2]
    metric_names = first_meta.get("MetricsOrder", [f"Metric_{i}" for i in range(expected_num_metrics)])
    logger.info(f"Expected tensor shape: ({expected_num_metrics}, {expected_num_rois}, {expected_num_rois})")
    logger.info(f"Metric names from first file: {metric_names}")
    if len(metric_names) != expected_num_metrics:
        logger.warning("Mismatch between number of metric names and data tensor's first dimension!")
        # Fallback if MetricsOrder is missing or incorrect
        metric_names = [f"Metric_{i}" for i in range(expected_num_metrics)]


    # --- Integrity Checks ---
    logger.info("--- Starting Integrity Checks ---")
    consistent_dims = True
    nan_found_count = 0
    inf_found_count = 0
    file_load_errors = 0
    
    # Store min/max for each metric type across all subjects
    # Initialize with extreme values
    global_min_max_per_metric = {name: [float('inf'), float('-inf')] for name in metric_names}


    for i, file_path in enumerate(pt_files):
        data_tensor, meta = load_pt_file(file_path)

        if data_tensor is None or meta is None:
            file_load_errors += 1
            continue
        
        all_metadata.append(meta)

        # Dimension check
        if data_tensor.shape != (expected_num_metrics, expected_num_rois, expected_num_rois):
            logger.warning(f"Dimension mismatch for {meta.get('SubjectID', file_path.name)}: "
                           f"Got {data_tensor.shape}, expected ({expected_num_metrics}, {expected_num_rois}, {expected_num_rois})")
            consistent_dims = False
        
        # NaN and Inf check
        if torch.isnan(data_tensor).any():
            logger.warning(f"NaNs found in data for {meta.get('SubjectID', file_path.name)}")
            nan_found_count += 1
        if torch.isinf(data_tensor).any():
            logger.warning(f"Infs found in data for {meta.get('SubjectID', file_path.name)}")
            inf_found_count += 1

        # Update min/max for each metric
        for metric_idx, metric_name in enumerate(metric_names):
            current_metric_data = data_tensor[metric_idx, :, :].numpy()
            # Exclude diagonal for min/max if it's always 0 and not representative
            # off_diag_values = current_metric_data[~np.eye(current_metric_data.shape[0],dtype=bool)]
            # For CVAE input, the whole matrix matters, so let's use all values or reconsider.
            # For now, using all values.
            min_val, max_val = np.min(current_metric_data), np.max(current_metric_data)
            
            if min_val < global_min_max_per_metric[metric_name][0]:
                global_min_max_per_metric[metric_name][0] = min_val
            if max_val > global_min_max_per_metric[metric_name][1]:
                global_min_max_per_metric[metric_name][1] = max_val
                
        # Store some examples for plotting
        if i < args.num_subjects_to_plot:
            example_data_to_plot.append((data_tensor, meta))

    logger.info("--- Integrity Checks Summary ---")
    logger.info(f"Total files processed: {len(pt_files) - file_load_errors}/{len(pt_files)}")
    if file_load_errors > 0:
        logger.warning(f"Number of files failed to load: {file_load_errors}")
    logger.info(f"Dimension consistency: {'OK' if consistent_dims else 'ISSUES FOUND - Check warnings'}")
    logger.info(f"Subjects with NaNs: {nan_found_count}")
    logger.info(f"Subjects with Infs: {inf_found_count}")
    
    logger.info("Global Min/Max values per metric:")
    for name, (min_v, max_v) in global_min_max_per_metric.items():
        logger.info(f"  {name}: Min = {min_v:.4f}, Max = {max_v:.4f}")


    # --- Metadata Analysis ---
    logger.info("--- Starting Metadata Analysis ---")
    if not all_metadata:
        logger.error("No metadata loaded, skipping metadata analysis and plotting.")
        return
        
    df_meta = pd.DataFrame(all_metadata)

    # Check for 'Group' column (essential for AD/CN classification)
    if "Group" not in df_meta.columns:
        logger.error("'Group' column not found in metadata. Cannot perform class-based analysis.")
    else:
        group_counts = df_meta["Group"].value_counts()
        logger.info("Class Distribution (AD/CN/Other):")
        logger.info(f"\n{group_counts.to_string()}")
        plot_class_distribution(group_counts, output_path=output_dir / "class_distribution.png")
        
        # Specific check for AD and CN counts
        ad_count = group_counts.get("AD", 0)
        cn_count = group_counts.get("CN", 0)
        other_count = group_counts.get("Other", 0) # Assuming 'Other' is the label for non-AD/CN
        unknown_count = len(df_meta) - (ad_count + cn_count + other_count) # Groups not explicitly AD, CN, or Other

        logger.info(f"AD subjects: {ad_count}")
        logger.info(f"CN subjects: {cn_count}")
        if other_count > 0:
            logger.info(f"'Other' labeled subjects: {other_count}")
        if unknown_count > 0:
            logger.warning(f"Subjects with group labels other than AD, CN, Other: {unknown_count}")

    # Check other potentially useful metadata columns
    if "NumROIsFinalDim" in df_meta.columns:
        roi_dim_counts = df_meta["NumROIsFinalDim"].value_counts()
        logger.info("Distribution of 'NumROIsFinalDim':")
        logger.info(f"\n{roi_dim_counts.to_string()}")
        if len(roi_dim_counts) > 1 or (expected_num_rois not in roi_dim_counts.index):
             logger.warning("Inconsistent 'NumROIsFinalDim' or mismatch with expected ROI count.")

    if "TimePointsForMetrics" in df_meta.columns:
        tp_counts = df_meta["TimePointsForMetrics"].value_counts().sort_index()
        logger.info("Distribution of 'TimePointsForMetrics':")
        logger.info(f"\n{tp_counts.to_string()}")


    # --- Visualization of Example Data ---
    logger.info("--- Starting Visualization of Example Data ---")
    if args.num_subjects_to_plot > 0 and not example_data_to_plot:
        logger.warning("Requested to plot example subjects, but no data was successfully loaded into examples (e.g. num_subjects_to_plot > files processed).")

    for data_tensor, meta in example_data_to_plot:
        subject_id = meta.get("SubjectID", "UnknownSubject")
        group = meta.get("Group", "Unk")
        logger.info(f"Plotting for Subject: {subject_id}, Group: {group}")

        # Determine which metrics to plot
        metrics_to_visualize_indices = []
        if args.metrics_to_plot: # User specified metrics
            for user_metric_name in args.metrics_to_plot:
                try:
                    metrics_to_visualize_indices.append(metric_names.index(user_metric_name))
                except ValueError:
                    logger.warning(f"Metric '{user_metric_name}' specified for plotting not found in available metrics: {metric_names}. Skipping it.")
        else: # Plot all metrics if none specified
            metrics_to_visualize_indices = list(range(expected_num_metrics))
            
        for metric_idx in metrics_to_visualize_indices:
            metric_name = metric_names[metric_idx]
            matrix_data = data_tensor[metric_idx, :, :].numpy()
            
            # Plot matrix
            matrix_plot_path = output_dir / f"matrix_{subject_id}_{group}_{metric_name.replace(' ','_')}.png"
            plot_connectivity_matrix(matrix_data, 
                                     title=f"Subject {subject_id} ({group})", 
                                     output_path=matrix_plot_path,
                                     metric_name=metric_name)
            
            # Plot distribution of off-diagonal values
            # The diagonal was set to 0.0 in the generation script.
            # For distribution, off-diagonal elements are more informative.
            off_diag_values = matrix_data[~np.eye(matrix_data.shape[0], dtype=bool)].flatten()
            dist_plot_path = output_dir / f"dist_{subject_id}_{group}_{metric_name.replace(' ','_')}.png"
            plot_value_distribution(off_diag_values,
                                    title=f"Off-Diagonal Values - Subject {subject_id} ({group})",
                                    output_path=dist_plot_path,
                                    metric_name=metric_name)
                                    
    logger.info(f"Visualizations (if any) saved to {output_dir}")
    logger.info("--- Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and check connectivity matrix tensors.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the .pt files from feature_extraction.py.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plots and reports.")
    parser.add_argument("--num_subjects_to_plot", type=int, default=3,
                        help="Number of example subjects to plot (matrices and distributions). Set to 0 to disable example plots.")
    parser.add_argument("--metrics_to_plot", type=str, nargs='*', default=None, # Example: --metrics_to_plot "Correlation_FisherZ" "NMI"
                        help="Specific metric names to plot for example subjects (e.g., 'Correlation_FisherZ'). If None, all metrics are plotted for example subjects.")
    
    args = parser.parse_args()
    main(args)