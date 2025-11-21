#!/usr/bin/env python3
"""
unified_analyser.py

Combines functionality from analyser.py and APD_plotter.py:
- Reads Export_DataAPD_run*.csv files (APD data)
- Reads DAQ_run*.csv files (temperature data)
- Detects throttle rise start and temperature rise start
- Aligns time points and creates unified CSV
- Plots aligned data with RC thermal model fitting
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re


# ---------------------------
# Helper Functions
# ---------------------------

def detect_throttle_rise_start(df_export, throttle_col="Throttle Duty", sampling_period=0.001, verification_window=5.0):
    """
    Detect when throttle duty starts to rise continuously.
    Checks that it doesn't go back to 0 for verification_window seconds.
    
    Returns: index where throttle rise starts
    """
    throttle = pd.to_numeric(df_export[throttle_col], errors='coerce')
    
    # Find first significant rise (threshold: > 2% to avoid noise)
    threshold = 2.0
    verification_points = int(verification_window / sampling_period)
    
    for i in range(len(throttle) - verification_points):
        if throttle.iloc[i] > threshold:
            # Check next verification_window seconds to ensure it doesn't drop back to 0
            window = throttle.iloc[i:i+verification_points]
            if (window > threshold).all() and window.min() > 0:
                return i
    
    # Fallback: return first non-zero value
    first_rise = (throttle > threshold).idxmax()
    return first_rise if pd.notna(first_rise) else 0


def detect_temperature_rise_start(df_daq, temp_col, time_col="Time_seconds"):
    """
    Detect when temperature starts to rise continuously.
    Looks for sustained increase over a short window.
    
    Returns: index where temperature rise starts
    """
    temp = pd.to_numeric(df_daq[temp_col], errors='coerce')
    time = pd.to_numeric(df_daq[time_col], errors='coerce')
    
    if len(temp) < 10:
        return 0
    
    # Calculate temperature gradient (rate of change)
    # Use a rolling window to smooth out noise
    window_size = min(10, len(temp) // 10)
    temp_diff = temp.diff(window_size)
    
    # Find first sustained positive gradient
    threshold = 0.01  # Minimum temperature increase per sample
    for i in range(window_size, len(temp_diff) - window_size):
        # Check if next few points show consistent increase
        if temp_diff.iloc[i] > threshold:
            # Verify it's a sustained rise (next 5 points also increasing)
            next_points = temp_diff.iloc[i:i+5]
            if (next_points > 0).sum() >= 3:  # At least 3 out of 5 increasing
                return i
    
    # Fallback: return index where temperature first exceeds initial + 0.5°C
    initial_temp = temp.iloc[:10].mean()
    first_rise = (temp > initial_temp + 0.5).idxmax()
    return first_rise if pd.notna(first_rise) else 0


def read_export_apd_file(file_path, sampling_period=0.001):
    """Read Export_DataAPD CSV file."""
    df = pd.read_csv(file_path)
    # Create time column based on index and sampling period
    df['Time (s)'] = df.index * sampling_period
    return df


def read_daq_file(file_path):
    """Read DAQ CSV file with metadata header."""
    metadata = {}
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Parse metadata
    for i, line in enumerate(lines[:10]):
        line_clean = line.strip().strip('"')
        if ':' in line_clean:
            key, value = line_clean.split(':', 1)
            metadata[key.strip()] = value.strip()
    
    # Find header row
    def _norm(s):
        return s.lstrip('\ufeff').strip().lower()
    
    header_row_idx = None
    for i, line in enumerate(lines):
        s = _norm(line)
        if 'sample' in s and ('date/time' in s or 'time' in s):
            header_row_idx = i
            break
    
    if header_row_idx is None:
        for i, line in enumerate(lines):
            s = _norm(line)
            if 'sample' in s and ',' in s:
                header_row_idx = i
                break
    
    if header_row_idx is None:
        raise ValueError(f"Could not find data header in {file_path}")
    
    # Read data
    df = pd.read_csv(file_path, skiprows=header_row_idx, encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.strip('"').str.strip()
    
    # Parse Time (s) column
    if 'Time (s)' in df.columns:
        def parse_time_s(val):
            try:
                s = str(val).strip().replace('"', '')
                parts = s.split(':')
                if len(parts) == 3:
                    h, m, sec = parts
                    return float(h) * 3600 + float(m) * 60 + float(sec)
                elif len(parts) == 2:
                    m, sec = parts
                    return float(m) * 60 + float(sec)
                else:
                    return float(s)
            except Exception:
                return np.nan
        df['Time_seconds'] = df['Time (s)'].apply(parse_time_s)
    
    # Convert numeric columns
    exclude = ['Sample', 'Date/Time', 'Time (s)', 'Time_seconds']
    numeric_cols = [c for c in df.columns if c not in exclude]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, metadata


def align_and_merge_data(df_export, df_daq, throttle_start_idx, temp_start_idx, 
                         temp_col, columns_to_plot, folder_name):
    """
    Align Export and DAQ data based on detected start points and create unified CSV.
    Only uses the overlapping time window between the two datasets.
    
    columns_to_plot can be:
    - List of strings: ["Power (W)", "Current (A)"]
    - List of dicts: [{"column": "Power (W)", "smooth": True, "method": "stratified", "window_size": 5.0}]
    - Mixed: ["Power (W)", {"column": "Current (A)", "smooth": True}]
    
    Returns: merged DataFrame with aligned times
    """
    # Extract data starting from detected points
    df_export_aligned = df_export.iloc[throttle_start_idx:].copy()
    df_daq_aligned = df_daq.iloc[temp_start_idx:].copy()
    
    # Reset time to start from 0 for both
    df_export_aligned['Time_aligned'] = df_export_aligned['Time (s)'] - df_export_aligned['Time (s)'].iloc[0]
    df_daq_aligned['Time_aligned'] = df_daq_aligned['Time_seconds'] - df_daq_aligned['Time_seconds'].iloc[0]
    
    # Find overlapping time window
    export_time_min = df_export_aligned['Time_aligned'].min()
    export_time_max = df_export_aligned['Time_aligned'].max()
    daq_time_min = df_daq_aligned['Time_aligned'].min()
    daq_time_max = df_daq_aligned['Time_aligned'].max()
    
    overlap_start = max(export_time_min, daq_time_min)
    overlap_end = min(export_time_max, daq_time_max)
    
    print(f"  Export time range: {export_time_min:.3f}s to {export_time_max:.3f}s (length: {export_time_max - export_time_min:.3f}s)")
    print(f"  DAQ time range: {daq_time_min:.3f}s to {daq_time_max:.3f}s (length: {daq_time_max - daq_time_min:.3f}s)")
    print(f"  Overlapping window: {overlap_start:.3f}s to {overlap_end:.3f}s (length: {overlap_end - overlap_start:.3f}s)")
    
    if overlap_end <= overlap_start:
        raise ValueError(f"No overlapping time window between Export and DAQ data!")
    
    # Filter to overlapping window
    df_export_overlap = df_export_aligned[
        (df_export_aligned['Time_aligned'] >= overlap_start) & 
        (df_export_aligned['Time_aligned'] <= overlap_end)
    ].copy()
    
    df_daq_overlap = df_daq_aligned[
        (df_daq_aligned['Time_aligned'] >= overlap_start) & 
        (df_daq_aligned['Time_aligned'] <= overlap_end)
    ].copy()
    
    # Use Export time grid (finer sampling) for merged data
    merged_data = {'Time (s)': df_export_overlap['Time_aligned'].values}
    time_values = df_export_overlap['Time_aligned'].values
    
    # Add Export columns (with optional smoothing)
    for col_spec in columns_to_plot:
        # Parse column specification
        if isinstance(col_spec, dict):
            col_name = col_spec.get('column', '')
            smooth = col_spec.get('smooth', False)
            method = col_spec.get('method', 'stratified')  # 'stratified' or 'rolling'
            window_size = col_spec.get('window_size', 5.0)  # seconds
            range_indices = col_spec.get('range', None)  # [start_idx, end_idx] or None
        else:
            col_name = col_spec
            smooth = False
            method = 'stratified'
            window_size = 5.0
            range_indices = None
        
        if col_name in df_export_overlap.columns:
            values = df_export_overlap[col_name].values
            
            # Apply smoothing if requested
            if smooth:
                if method == 'stratified':
                    values = smooth_stratified_average(time_values, values, window_size, range_indices)
                    range_str = f" (range: {range_indices})" if range_indices else ""
                    print(f"  Applied stratified averaging to '{col_name}' (window: {window_size}s{range_str})")
                elif method == 'rolling':
                    values = smooth_rolling_average(time_values, values, window_size, range_indices)
                    range_str = f" (range: {range_indices})" if range_indices else ""
                    print(f"  Applied rolling average to '{col_name}' (window: {window_size}s{range_str})")
                else:
                    print(f"  Warning: Unknown smoothing method '{method}', using raw data")
            
            merged_data[col_name] = values
    
    # Add temperature from DAQ (interpolated onto Export time grid)
    if temp_col in df_daq_overlap.columns:
        # Interpolate temperature onto Export time grid
        temp_interp = np.interp(
            df_export_overlap['Time_aligned'].values,
            df_daq_overlap['Time_aligned'].values,
            df_daq_overlap[temp_col].values
        )
        merged_data[temp_col] = temp_interp
    
    df_merged = pd.DataFrame(merged_data)
    
    # Save unified CSV
    output_file = f"{folder_name}_unified.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"  Saved unified CSV: {output_file} ({len(df_merged)} data points)")
    
    return df_merged


def rc_thermal_model(t, T_inf, T_0, tau):
    """RC thermal circuit model: T(t) = T_inf - (T_inf - T_0) * exp(-t/tau)"""
    return T_inf - (T_inf - T_0) * np.exp(-t / tau)


def smooth_stratified_average(time, values, window_size=5.0, range_indices=None):
    """
    Smooth data using stratified averaging (binned averaging).
    
    Divides the time axis into bins of size window_size and averages values within each bin.
    Returns smoothed values at the original time points (using interpolation).
    
    Parameters:
    -----------
    time : array-like
        Time values
    values : array-like
        Data values to smooth
    window_size : float
        Time window size in seconds for each bin
    range_indices : tuple or list, optional
        (start_idx, end_idx) to limit smoothing to a specific range of data points.
        If None, smooths the entire dataset.
        
    Returns:
    --------
    smoothed_values : numpy array
        Smoothed values at original time points (only smoothed within range if specified)
    """
    time = np.array(time)
    values = np.array(values)
    smoothed = values.copy()
    
    # Determine the range to smooth
    if range_indices is not None:
        # Handle single-element range (start only) - means "from start to end"
        if len(range_indices) == 1:
            start_idx = max(0, int(range_indices[0]))
            end_idx = len(time)
        else:
            start_idx, end_idx = range_indices[0], range_indices[1]
            start_idx = max(0, int(start_idx))
            end_idx = min(len(time), int(end_idx))
        if start_idx >= end_idx:
            return values
        range_mask = np.zeros(len(time), dtype=bool)
        range_mask[start_idx:end_idx] = True
    else:
        range_mask = np.ones(len(time), dtype=bool)
        start_idx = 0
        end_idx = len(time)
    
    # Extract data within the range
    time_range = time[range_mask]
    values_range = values[range_mask]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(time_range) | np.isnan(values_range))
    if not valid_mask.any():
        return values
    
    time_valid = time_range[valid_mask]
    values_valid = values_range[valid_mask]
    
    # If window_size is <= 1.0, return original data (no smoothing)
    # This ensures that window_size=1 preserves the original data exactly
    if window_size <= 1.0:
        return values
    
    # Additional check: if window_size is very small relative to data spacing, return original data
    if len(time_valid) > 1:
        time_sorted = np.sort(time_valid)
        typical_spacing = np.median(np.diff(time_sorted))
        # If window_size is less than 5x the typical spacing, return original data
        if window_size < 5 * typical_spacing:
            return values
    
    # Create bins
    time_min = time_valid.min()
    time_max = time_valid.max()
    bins = np.arange(time_min, time_max + window_size, window_size)
    
    # Bin the data and compute averages
    bin_indices = np.digitize(time_valid, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)  # Ensure valid indices
    
    bin_centers = []
    bin_averages = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_averages.append(np.mean(values_valid[mask]))
    
    if len(bin_centers) == 0:
        return values
    
    bin_centers = np.array(bin_centers)
    bin_averages = np.array(bin_averages)
    
    # For each original time point, find which bin it belongs to and use that bin's average
    # This preserves the data better than interpolation
    time_range_valid = time_range[valid_mask]
    smoothed_range = np.zeros_like(time_range_valid)
    
    for i, t in enumerate(time_range_valid):
        # Find which bin this time point belongs to
        bin_idx = np.digitize([t], bins)[0] - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_averages) - 1)
        smoothed_range[i] = bin_averages[bin_idx]
    
    # Map back to original indices
    # Get the indices in the original array that correspond to the range
    range_indices_array = np.where(range_mask)[0]  # All indices in range
    valid_range_indices = range_indices_array[valid_mask]  # Only valid indices in range
    
    # Map smoothed values back to original array positions
    for i, orig_idx in enumerate(valid_range_indices):
        smoothed[orig_idx] = smoothed_range[i]
    
    return smoothed


def smooth_rolling_average(time, values, window_size=5.0, range_indices=None):
    """
    Smooth data using rolling average based on time window.
    
    Parameters:
    -----------
    time : array-like
        Time values
    values : array-like
        Data values to smooth
    window_size : float
        Time window size in seconds for rolling average
    range_indices : tuple or list, optional
        (start_idx, end_idx) to limit smoothing to a specific range of data points.
        If None, smooths the entire dataset.
        
    Returns:
    --------
    smoothed_values : numpy array
        Smoothed values at original time points (only smoothed within range if specified)
    """
    time = np.array(time)
    values = np.array(values)
    smoothed = values.copy()
    
    # Determine the range to smooth
    if range_indices is not None:
        # Handle single-element range (start only) - means "from start to end"
        if len(range_indices) == 1:
            start_idx = max(0, int(range_indices[0]))
            end_idx = len(time)
        else:
            start_idx, end_idx = range_indices[0], range_indices[1]
            start_idx = max(0, int(start_idx))
            end_idx = min(len(time), int(end_idx))
        if start_idx >= end_idx:
            return values
        range_mask = np.zeros(len(time), dtype=bool)
        range_mask[start_idx:end_idx] = True
    else:
        range_mask = np.ones(len(time), dtype=bool)
        start_idx = 0
        end_idx = len(time)
    
    # Extract data within the range
    time_range = time[range_mask]
    values_range = values[range_mask]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(time_range) | np.isnan(values_range))
    if not valid_mask.any():
        return values
    
    time_valid = time_range[valid_mask]
    values_valid = values_range[valid_mask]
    
    # If window_size is <= 1.0, return original data (no smoothing)
    # This ensures that window_size=1 preserves the original data exactly
    if window_size <= 1.0:
        return values
    
    # For time-based rolling, compute manually within the range
    range_indices_array = np.where(range_mask)[0]
    for i, idx in enumerate(range_indices_array):
        if valid_mask[i]:
            # Find all points within window_size/2 of current time
            time_center = time_range[i]
            mask = (time_valid >= time_center - window_size/2) & (time_valid <= time_center + window_size/2)
            if mask.sum() > 0:
                smoothed[idx] = np.mean(values_valid[mask])
            else:
                smoothed[idx] = values_range[i]
        else:
            smoothed[idx] = values_range[i]
    
    return smoothed


# ---------------------------
# Main Processing
# ---------------------------

if __name__ == "__main__":
    # ======================================================================
    # EDIT THESE VARIABLES:
    # ======================================================================
    # 1. Multiple files: List of folder names
    folder_names = ["Run2-Throttle100", "Run13-Throttle8"] #"Run3-Throttle100"]
    #folder_names = ["Run8-Throttle100", "Run14-Throttle9" ] #"Run3-Throttle100"]
    

    # 2. Columns from Export_DataAPD files to plot (dictionary: folder_name -> list of columns)
    # 
    # You can specify columns in three ways:
    # - Simple string: "Power (W)" - plots raw data
    # - Dictionary with smoothing: {"column": "Power (W)", "smooth": True, "method": "stratified", "window_size": 5.0}
    #   - "smooth": True/False - enable/disable smoothing
    #   - "method": "stratified" (binned averaging) or "rolling" (rolling average)
    #   - "window_size": time window in seconds (default: 5.0)
    #   - "range": [start_idx, end_idx] or [start_idx] - optional, data point indices to smooth and plot
    #     Examples: [20, 1020] means points 20 to 1020, [20] means from point 20 to the end
    #     If specified, only this range will be plotted for this column (data keeps original time positions)
    #     Each column can have its own range, allowing different ranges for different columns
    #
    # Example with smoothing and range (per-column):
    # COLUMNS_TO_PLOT = {
    #     "Run2-Throttle100": [
    #         {"column": "Power (W)", "smooth": True, "method": "stratified", "window_size": 5.0, "range": [20, 1020]},
    #         {"column": "Current (A)", "smooth": True, "method": "stratified", "window_size": 5.0, "range": [50, 1050]}  # Different range!
    #     ],
    # }
    #
    # Example with smoothing (full dataset):
    COLUMNS_TO_PLOT = {
        "Run2-Throttle100": [
            {"column": "Power (W)", "smooth": True, "method": "stratified", "window_size": 1, "range": [20, 2000]}
        ],
        "Run13-Throttle8": [
            {"column": "Power (W)", "smooth": True, "method": "stratified", "window_size": 1, "range": [150, 500]}
        ],
    }
    
 #   Example without smoothing (commented out):
    # COLUMNS_TO_PLOT = {
    #     "Run2-Throttle100": ["Power (W)"],
    #     "Run13-Throttle8": ["Power (W)"],
    # }
    
    # 3. Temperature parameter from DAQ files
    temperature_param = "Winding Temp (°C)" 
    # For the ARES stator - this corresponds to Winding1 (°C) hooked up to channel 7. The other winding IS cooler, but partly because the thermocouple keeps coming off - it was put at a different location.
    # For the PH3 in Run8 - the header is swapped to get a plot!

    # 4. Sampling period for Export_DataAPD files (seconds)
    SAMPLING_PERIOD = 0.1
    
    # 5. Time window to verify throttle rise (seconds) - will automatically adjust the number of datapoints parsed to keep a consistent - check that throttle doesn't go back to 0
    THROTTLE_VERIFICATION_WINDOW = 5.0
    
    # 6. Plot title
    plot_title = "Aligned Temperature and APD Data"
    
    # 7. Curve fitting time range (in seconds, relative to aligned time)
    # Dictionary with keys: "fit_start_1", "fit_start_2", etc. for each folder (by index)
    fit_start_seconds = {
        "fit_start_1": 10,  # For first folder in folder_names list
        "fit_start_2": 20,  # For second folder in folder_names list
       # "fit_start_3": 10,  # For third folder in folder_names list
    #    "fit_start_4": 15,  # For third folder in folder_names list
        # Add more as needed: "fit_start_4", etc.
    }
    
    # Dictionary with keys: "fit_end_1", "fit_end_2", etc. for each folder (by index)
    fit_end_seconds = {
        "fit_end_1": 300,  # For first folder in folder_names list
        "fit_end_2": 150,     # For second folder in folder_names list
    #    "fit_end_3": 190,  # For third folder in folder_names list
        # Add more as needed: "fit_end_4", etc.
    }
    
    # 8. Prediction time limits (in seconds) - how far to extend the fitted curve
    # Dictionary with keys: "prediction_limit_1", "prediction_limit_2", etc. for each folder (by index)
    prediction_time_limit_seconds = {
        "prediction_limit_1": 300,  # For first folder in folder_names list
        "prediction_limit_2": 300,   # For second folder in folder_names list
        # Add more as needed: "prediction_limit_4", etc.
        # Or set to None for a folder to use default (1.5x max time)
    }
    # ======================================================================
    
    base_path = Path.cwd()
    all_merged_data = {}
    fitted_functions = []
    
    # Create mapping from folder name to index
    folder_to_index = {}
    for idx, folder_name in enumerate(folder_names, start=1):
        folder_to_index[folder_name] = idx
    
    # Process each folder
    for folder_name in folder_names:
        folder_path = base_path / folder_name
        
        # Find files
        export_file = None
        daq_file = None
        
        for csv_file in folder_path.glob("*.csv"):
            if csv_file.name.startswith("Export_DataAPD"):
                export_file = csv_file
            elif csv_file.name.startswith("DAQ_"):
                daq_file = csv_file
        
        if export_file is None:
            print(f"Warning: No Export_DataAPD file found in {folder_name}")
            continue
        if daq_file is None:
            print(f"Warning: No DAQ file found in {folder_name}")
            continue
        
        print(f"\nProcessing {folder_name}...")
        print(f"  Export file: {export_file.name}")
        print(f"  DAQ file: {daq_file.name}")
        
        # Read files
        try:
            df_export = read_export_apd_file(export_file, sampling_period=SAMPLING_PERIOD)
            df_daq, metadata = read_daq_file(daq_file)
        except Exception as e:
            print(f"Error reading files: {e}")
            continue
        
        # Use the temperature parameter (same for all folders)
        
        # Detect start points
        throttle_start_idx = detect_throttle_rise_start(df_export, sampling_period=SAMPLING_PERIOD, 
                                                         verification_window=THROTTLE_VERIFICATION_WINDOW)
        temp_start_idx = detect_temperature_rise_start(df_daq, temperature_param)
        
        print(f"  Throttle rise starts at index: {throttle_start_idx} (time: {throttle_start_idx * SAMPLING_PERIOD:.3f}s)")
        print(f"  Temperature rise starts at index: {temp_start_idx} (time: {df_daq['Time_seconds'].iloc[temp_start_idx]:.3f}s)")
        
        # Align and merge
        try:
            # Get columns to plot for this folder
            columns_for_folder = COLUMNS_TO_PLOT.get(folder_name, COLUMNS_TO_PLOT.get(list(COLUMNS_TO_PLOT.keys())[0], []))
            df_merged = align_and_merge_data(df_export, df_daq, throttle_start_idx, temp_start_idx,
                                            temperature_param, columns_for_folder, folder_name)
            all_merged_data[folder_name] = df_merged
        except Exception as e:
            print(f"Error aligning data: {e}")
            continue
    
    if not all_merged_data:
        print("No data processed. Check file names and paths.")
    else:
        # Create plot with dual y-axes
        fig, ax = plt.subplots(figsize=(14, 8))
        ax2 = ax.twinx()  # Second y-axis for APD data
        
        # Plot data from each folder
        for folder_name, df_merged in all_merged_data.items():
            folder_idx = folder_to_index.get(folder_name)
            
            # Plot temperature on left y-axis
            if temperature_param in df_merged.columns:
                ax.plot(df_merged['Time (s)'], df_merged[temperature_param],
                       label=f"{folder_name} - {temperature_param}", 
                       linewidth=2, marker='o', markersize=3)
            
            # Plot Export columns on right y-axis
            columns_for_folder = COLUMNS_TO_PLOT.get(folder_name, COLUMNS_TO_PLOT.get(list(COLUMNS_TO_PLOT.keys())[0], []))
            for col_spec in columns_for_folder:
                # Parse column specification
                if isinstance(col_spec, dict):
                    col_name = col_spec.get('column', '')
                    smooth = col_spec.get('smooth', False)
                    range_indices = col_spec.get('range', None)  # [start_idx, end_idx] or None
                    label_suffix = " (smoothed)" if smooth else ""
                else:
                    col_name = col_spec
                    range_indices = None
                    label_suffix = ""
                
                if col_name in df_merged.columns:
                    # Extract data for this column
                    time_data = df_merged['Time (s)'].values
                    col_data = df_merged[col_name].values
                    
                    # Filter to specified range if provided
                    if range_indices is not None:
                        # Handle single-element range (start only) - means "from start to end"
                        if len(range_indices) == 1:
                            start_idx = max(0, int(range_indices[0]))
                            end_idx = len(df_merged)
                        else:
                            start_idx, end_idx = range_indices[0], range_indices[1]
                            start_idx = max(0, int(start_idx))
                            end_idx = min(len(df_merged), int(end_idx))
                        
                        if start_idx < end_idx:
                            # Python slicing is exclusive of end, so use end_idx+1 to include the end index
                            # But cap it at the array length to avoid index errors
                            end_slice = min(end_idx + 1, len(time_data))
                            time_data = time_data[start_idx:end_slice]
                            col_data = col_data[start_idx:end_slice]
                            # Keep original time positions (don't shift to t=0)
                            # The range just filters which points to show
                            if len(range_indices) == 1:
                                label_suffix += f" [range: {start_idx}-end]"
                            else:
                                label_suffix += f" [range: {start_idx}-{end_idx}]"
                    
                    ax2.plot(time_data, col_data,
                           label=f"{folder_name} - {col_name}{label_suffix}", 
                           linewidth=1.5, linestyle='--', alpha=0.7)
            
            # RC Thermal Model Fitting
            if temperature_param in df_merged.columns:
                valid_data = df_merged[
                    (df_merged[temperature_param].notna()) & 
                    (df_merged['Time (s)'].notna())
                ].copy()
                
                if len(valid_data) > 3:
                    # Get fit times
                    fit_start = None
                    fit_end = None
                    if folder_idx is not None:
                        fit_start_key = f"fit_start_{folder_idx}"
                        fit_end_key = f"fit_end_{folder_idx}"
                        fit_start = fit_start_seconds.get(fit_start_key, None)
                        fit_end = fit_end_seconds.get(fit_end_key, None)
                    
                    if fit_start is None:
                        fit_start = valid_data['Time (s)'].min()
                    if fit_end is None:
                        fit_end = valid_data['Time (s)'].max()
                    
                    # Extract fitting data
                    fit_mask = (valid_data['Time (s)'] >= fit_start) & (valid_data['Time (s)'] <= fit_end)
                    fit_data = valid_data[fit_mask].copy()
                    
                    if len(fit_data) > 3:
                        t_fit = fit_data['Time (s)'].values - fit_data['Time (s)'].iloc[0]
                        T_fit = fit_data[temperature_param].values
                        
                        # Initial guesses
                        T_0_est = T_fit[0]
                        T_inf_est = T_fit[-1] if len(T_fit) > 1 else T_0_est + 10
                        tau_est = (fit_data['Time (s)'].iloc[-1] - fit_data['Time (s)'].iloc[0]) / 3
                        
                        try:
                            popt, pcov = curve_fit(rc_thermal_model, t_fit, T_fit,
                                                  p0=[T_inf_est, T_0_est, tau_est],
                                                  maxfev=5000)
                            T_inf, T_0, tau = popt
                            time_offset = fit_data['Time (s)'].iloc[0]
                            
                            fitted_functions.append({
                                'folder': folder_name,
                                'T_inf': T_inf,
                                'T_0': T_0,
                                'tau': tau,
                                'time_offset': time_offset
                            })
                            
                            # Print fitted parameters
                            print(f"\n{folder_name} - Fitted RC Thermal Model:")
                            print(f"  T(t) = {T_inf:.2f} - ({T_inf:.2f} - {T_0:.2f}) * exp(-(t-{time_offset:.1f})/{tau:.2f})")
                            print(f"  Parameters: T_inf={T_inf:.2f}°C, T_0={T_0:.2f}°C, tau={tau:.2f}s")
                            
                            # Prediction range
                            pred_end = None
                            if folder_idx is not None:
                                pred_limit_key = f"prediction_limit_{folder_idx}"
                                pred_end = prediction_time_limit_seconds.get(pred_limit_key, None)
                            
                            if pred_end is None:
                                pred_end = valid_data['Time (s)'].max() * 1.5
                            
                            t_pred = np.linspace(0, pred_end - time_offset, 200)
                            T_pred = rc_thermal_model(t_pred, T_inf, T_0, tau)
                            t_pred_absolute = t_pred + time_offset
                            
                            # Get the color from the last temperature line for this folder
                            temp_line_color = None
                            for line in ax.lines:
                                if folder_name in line.get_label() and temperature_param in line.get_label():
                                    temp_line_color = line.get_color()
                                    break
                            
                            ax.plot(t_pred_absolute, T_pred, '--',
                                   label=f"{folder_name} (fitted)", 
                                   linewidth=2, alpha=0.8, color=temp_line_color if temp_line_color else None)
                            
                        except Exception as e:
                            print(f"\nWarning: Curve fitting failed for {folder_name}: {e}")
        
        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        temp_unit = temperature_param.split("(")[-1].split(")")[0] if "(" in temperature_param else "°C"
        ax.set_ylabel(f'Temperature ({temp_unit})', fontsize=12, color='blue')
        ax2.set_ylabel('APD Data (Current/Power/Throttle)', fontsize=12, color='red')
        
        # Color the y-axis labels to match the data
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9, ncol=2)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\nPlot generated with {len(all_merged_data)} folders")
        if fitted_functions:
            print(f"Fitted {len(fitted_functions)} curves")

