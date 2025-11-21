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
    
    # Add Export columns
    for col in columns_to_plot:
        if col in df_export_overlap.columns:
            merged_data[col] = df_export_overlap[col].values
    
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
    COLUMNS_TO_PLOT = {
        "Run2-Throttle100": ["Power (W)"],
 #       "Run8-Throttle100": ["Power (W)"],
        "Run13-Throttle8": ["Power (W)"],
  #      "Run14-Throttle9": ["Power (W)"],
        #"Run3-Throttle100": ["Power (W)"],
    }
    
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
            for col in columns_for_folder:
                if col in df_merged.columns:
                    ax2.plot(df_merged['Time (s)'], df_merged[col],
                           label=f"{folder_name} - {col}", 
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

