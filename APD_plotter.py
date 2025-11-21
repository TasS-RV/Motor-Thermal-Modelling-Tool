import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================================
# USER-CONFIGURE SETTINGS (EDIT THESE)
# =========================================================

# Input CSV file
run_num = 1
CSV_FILE = "./Run1-Throttle37/Export_DataAPD_run1.csv"

# Stage 1: Which columns to plot (as they appear in the CSV header)
COLUMNS_TO_PLOT = [
    # Example – update for your file:
    "Phase Current (A)",
    "Power (W)",
    "Temperature (C)",
]

# Stage 1: Raw index limits to initially inspect
# Set to None to view entire dataset
RAW_START_INDEX = None
RAW_END_INDEX   = None

# Stage 2 (after you inspect Stage 1 plots):
# Final selected index chunk for exporting & final plotting
FINAL_START_INDEX = 5000
FINAL_END_INDEX   = 15000   # Use None to go to end

# Sampling interval (seconds)
# Example: if raw data = 1ms sampling, this = 0.001
SAMPLING_PERIOD = 0.001

# Output CSV location
OUTPUT_CSV = "processed_output.csv"

# =========================================================
# LOAD DATA
# =========================================================
print(f"Loading {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)

# Ensure requested columns exist
missing = [c for c in COLUMNS_TO_PLOT if c not in df.columns]
if missing:
    raise ValueError(f"These columns are not in the CSV: {missing}")

# =========================================================
# STAGE 1 – Quick preview of raw dataset
# =========================================================
print("\n=== STAGE 1: Plotting raw data for visual inspection ===")

raw_df = df.copy()

# Slice raw data
if RAW_START_INDEX is not None or RAW_END_INDEX is not None:
    raw_df = raw_df.iloc[
        RAW_START_INDEX if RAW_START_INDEX is not None else 0 :
        RAW_END_INDEX if RAW_END_INDEX is not None else len(df)
    ]

# Plot stage-1 preview
plt.figure(figsize=(12, 6))
for col in COLUMNS_TO_PLOT:
    plt.plot(raw_df.index, raw_df[col], label=col)

plt.title("Stage 1 – Raw Data Preview")
plt.xlabel("Row Index")
plt.ylabel("Value")
plt.legend()
plt.show()

print("Review this plot and adjust FINAL_START_INDEX and FINAL_END_INDEX")
print("Then re-run the script.")

# =========================================================
# STAGE 2 – Extract final range, compute time column
# =========================================================
print("\n=== STAGE 2: Extracting final selection ===")

final_df = df.copy()

# Final range slicing
final_df = final_df.iloc[
    FINAL_START_INDEX : FINAL_END_INDEX if FINAL_END_INDEX is not None else len(df)
]

# Keep only desired columns
final_df = final_df[COLUMNS_TO_PLOT]

# Add time column based on index offset
num_points = len(final_df)
final_df["Time (s)"] = (final_df.index - FINAL_START_INDEX) * SAMPLING_PERIOD

# =========================================================
# SAVE NEW CSV
# =========================================================
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Exported processed CSV to: {os.path.abspath(OUTPUT_CSV)}")

# =========================================================
# FINAL PLOT
# =========================================================
plt.figure(figsize=(12, 6))
for col in COLUMNS_TO_PLOT:
    plt.plot(final_df["Time (s)"], final_df[col], label=col)

plt.title("Stage 2 – Final Processed Data")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
