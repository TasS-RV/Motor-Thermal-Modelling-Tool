import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm

# ==========================================
# 🎛️ CONTROL PANEL
# ==========================================

# --- Experiment Description ---
EXPERIMENT_DESCRIPTION = """
Colder to actual conditions: 85 C ambient, 8 W heater, 85 C oil from just motor side. Assuming infinite oil thermal mass, but a gentler experiment case.
"""

# --- New Physics Toggles ---
OIL_IS_FINITE = False        # True = Oil heats up over time. False = Infinite constant temp source.
OIL_PLACEMENT = "RIGHT"       # Options: "ALL", "LEFT", "RIGHT", "TOP", "BOTTOM"

# --- Snapshot Settings ---
SAVE_SNAPSHOTS      = True           
SNAPSHOT_INTERVAL_S = 3.0          # Save image every 5 minutes
SNAPSHOT_FOLDER     = "Test - true milder condition" 

# --- Geometry [m] ---
space_length = 0.094 
L_air_inner    = 0.100       
t_insulation   = 0.005       
t_oil_margin   = 0.050       
D_depth        = space_length - (0.015 - 0.016 - 0.008) 

# --- Heater Source ---
L_heater_block = 0.030      
Q_input_watts  = 8.0       

# --- Temperatures [K] ---
T_oil_setpoint = 273.15 + 85.0   # 170°C
T_ambient      = 273.15 + 85.0          # 20°C (Room temp for non-oil sides)
T_initial      = 293.15          # 20°C (Starting temp of everything - always room temp)

# --- Simulation Settings ---
Resolution     = 140         
Time_Total     = 36       
Animation_Speedup = 50.0    # Playback speed

# --- Materials ---
# Note: Added 'ambient' for the empty air space in single-sided mode
mat_ins     = {'k': 0.02,   'rho': 150.0,  'cp': 1000.0} # Aerogel
mat_air_int = {'k': 0.026,  'rho': 1.225,  'cp': 1005.0} # Internal Air
mat_air_amb = {'k': 0.026,  'rho': 1.225,  'cp': 1005.0} # External Ambient Air
mat_heat    = {'k': 160.0,  'rho': 2700.0, 'cp': 900.0}  # Aluminum Heater - while somewhat unrealistic, a small aluminium heatsink WILL be required for the heatsink. This somewhat encapsulates a bit of the volume of the faraday cage as well
mat_oil     = {'k': 0.15,   'rho': 800.0,  'cp': 2000.0} # Oil

# ==========================================
# ⚙️ SETUP
# ==========================================

if SAVE_SNAPSHOTS and not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)

# Function to save experiment log
def save_experiment_log(folder_path, description, params_dict):
    """Save experiment description and parameters to a text file in the snapshot folder."""
    log_file = os.path.join(folder_path, "experiment_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT LOG\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("EXPERIMENT DESCRIPTION:\n")
        f.write("-" * 70 + "\n")
        f.write(description.strip() + "\n\n")
        
        f.write("KEY PARAMETERS:\n")
        f.write("-" * 70 + "\n")
        
        # Geometry Section
        f.write("\n--- GEOMETRY ---\n")
        f.write(f"Space Length: {params_dict['space_length']*1000:.2f} mm\n")
        f.write(f"Air Inner (L_air_inner): {params_dict['L_air_inner']*1000:.2f} mm\n")
        f.write(f"Insulation Thickness (t_insulation): {params_dict['t_insulation']*1000:.2f} mm\n")
        f.write(f"Oil Margin (t_oil_margin): {params_dict['t_oil_margin']*1000:.2f} mm\n")
        f.write(f"Depth (D_depth): {params_dict['D_depth']*1000:.2f} mm\n")
        f.write(f"Heater Block Size (L_heater_block): {params_dict['L_heater_block']*1000:.2f} mm\n")
        f.write(f"Grid Resolution: {params_dict['Resolution']}×{params_dict['Resolution']}\n")
        f.write(f"Grid Spacing (dx): {params_dict['dx']*1000:.4f} mm\n")
        f.write(f"Total Domain Size: {params_dict['L_total']*1000:.2f} mm\n")
        
        # Temperatures Section
        f.write("\n--- TEMPERATURES ---\n")
        f.write(f"Oil Setpoint: {params_dict['T_oil_setpoint']:.2f} K ({params_dict['T_oil_setpoint']-273.15:.2f}°C)\n")
        f.write(f"Ambient Temperature: {params_dict['T_ambient']:.2f} K ({params_dict['T_ambient']-273.15:.2f}°C)\n")
        f.write(f"Initial Temperature: {params_dict['T_initial']:.2f} K ({params_dict['T_initial']-273.15:.2f}°C)\n")
        
        # Oil Configuration
        f.write("\n--- OIL CONFIGURATION ---\n")
        f.write(f"Oil Placement: {params_dict['OIL_PLACEMENT']}\n")
        f.write(f"Oil Type: {'FINITE (Heats up over time)' if params_dict['OIL_IS_FINITE'] else 'INFINITE (Constant temperature)'}\n")
        
        # Heat Source
        f.write("\n--- HEAT SOURCE ---\n")
        f.write(f"Heater Power: {params_dict['Q_input_watts']:.2f} W\n")
        f.write(f"Heater Volume: {params_dict['vol_heater']*1e9:.2f} mm³\n")
        f.write(f"Volumetric Heat Generation: {params_dict['Q_volumetric']:.2e} W/m³\n")
        
        # Materials
        f.write("\n--- MATERIAL PROPERTIES ---\n")
        f.write("Insulation (Aerogel):\n")
        f.write(f"  k = {params_dict['mat_ins']['k']:.4f} W/mK\n")
        f.write(f"  ρ = {params_dict['mat_ins']['rho']:.1f} kg/m³\n")
        f.write(f"  cp = {params_dict['mat_ins']['cp']:.1f} J/kgK\n")
        f.write("Internal Air:\n")
        f.write(f"  k = {params_dict['mat_air_int']['k']:.4f} W/mK\n")
        f.write(f"  ρ = {params_dict['mat_air_int']['rho']:.3f} kg/m³\n")
        f.write(f"  cp = {params_dict['mat_air_int']['cp']:.1f} J/kgK\n")
        f.write("Oil:\n")
        f.write(f"  k = {params_dict['mat_oil']['k']:.4f} W/mK\n")
        f.write(f"  ρ = {params_dict['mat_oil']['rho']:.1f} kg/m³\n")
        f.write(f"  cp = {params_dict['mat_oil']['cp']:.1f} J/kgK\n")
        f.write("Heater (Aluminum):\n")
        f.write(f"  k = {params_dict['mat_heat']['k']:.1f} W/mK\n")
        f.write(f"  ρ = {params_dict['mat_heat']['rho']:.1f} kg/m³\n")
        f.write(f"  cp = {params_dict['mat_heat']['cp']:.1f} J/kgK\n")
        
        # Simulation Settings
        f.write("\n--- SIMULATION SETTINGS ---\n")
        f.write(f"Total Simulation Time: {params_dict['Time_Total']:.1f} s ({params_dict['Time_Total']/60:.2f} min, {params_dict['Time_Total']/3600:.3f} hours)\n")
        f.write(f"Time Step (dt): {params_dict['dt']:.6f} s\n")
        f.write(f"Number of Steps: {params_dict['n_steps']:,}\n")
        f.write(f"Snapshot Interval: {params_dict['SNAPSHOT_INTERVAL_S']:.1f} s ({params_dict['SNAPSHOT_INTERVAL_S']/60:.2f} min)\n")
        f.write(f"Animation Speedup: {params_dict['Animation_Speedup']:.1f}x\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Log file generated automatically by simulation.\n")
    
    print(f"Experiment log saved to: {log_file}")

# Prepare parameters dictionary for logging (will be populated after setup)
experiment_params = {}

# 1. Grid Generation
L_total = L_air_inner + 2*t_insulation + 2*t_oil_margin
dx = L_total / Resolution
dy = dx
x = np.linspace(0, L_total, Resolution)
y = np.linspace(0, L_total, Resolution)
X, Y = np.meshgrid(x, y)

# 2. Geometry Masks
center = L_total / 2
dist_x = np.abs(X - center)
dist_y = np.abs(Y - center)

# Core Components
mask_heater   = (dist_x < L_heater_block/2) & (dist_y < L_heater_block/2)
limit_air     = L_air_inner / 2
mask_air_zone = (dist_x < limit_air) & (dist_y < limit_air)
mask_air      = mask_air_zone & (~mask_heater)
limit_ins     = limit_air + t_insulation
mask_ins_zone = (dist_x < limit_ins) & (dist_y < limit_ins)
mask_ins      = mask_ins_zone & (~mask_air_zone)

# 3. Oil Placement Logic
# Define where oil exists based on OIL_PLACEMENT
mask_oil_region = np.zeros_like(X, dtype=bool)

if OIL_PLACEMENT == "ALL":
    mask_oil_region = ~mask_ins_zone # Everywhere outside insulation
elif OIL_PLACEMENT == "LEFT":
    mask_oil_region = (X < t_oil_margin)
elif OIL_PLACEMENT == "RIGHT":
    mask_oil_region = (X > L_total - t_oil_margin)
elif OIL_PLACEMENT == "BOTTOM":
    mask_oil_region = (Y < t_oil_margin)
elif OIL_PLACEMENT == "TOP":
    mask_oil_region = (Y > L_total - t_oil_margin)

# Ensure Oil doesn't clip into the box
mask_oil = mask_oil_region & (~mask_ins_zone)

# 4. Ambient Air Logic
# Anywhere that is NOT heater, NOT internal air, NOT insulation, and NOT oil... is Ambient Air
mask_ambient = ~(mask_heater | mask_air | mask_ins | mask_oil)

# 5. Apply Material Properties
K   = np.zeros_like(X); Rho = np.zeros_like(X); Cp  = np.zeros_like(X)

def apply_mat(mask, mat):
    K[mask] = mat['k']; Rho[mask] = mat['rho']; Cp[mask] = mat['cp']

apply_mat(mask_ambient, mat_air_amb)
apply_mat(mask_oil, mat_oil)
apply_mat(mask_ins, mat_ins)
apply_mat(mask_air, mat_air_int)
apply_mat(mask_heater, mat_heat)

# 6. Stability Check
vol_heater = np.sum(mask_heater) * dx * dy * D_depth
Q_volumetric = Q_input_watts / vol_heater
Alpha = K / (Rho * Cp)
alpha_max = np.max(Alpha) # Check everywhere
dt = (0.8 * dx**2) / (4 * alpha_max)

print(f"--- SIMULATION CONFIG ---")
print(f"Oil Type: {'FINITE (Heats up)' if OIL_IS_FINITE else 'INFINITE (Constant Temp)'}")
print(f"Oil Side: {OIL_PLACEMENT}")
print(f"Time Step: {dt:.5f} s")

# Collect all parameters for logging
experiment_params = {
    'space_length': space_length,
    'L_air_inner': L_air_inner,
    't_insulation': t_insulation,
    't_oil_margin': t_oil_margin,
    'D_depth': D_depth,
    'L_heater_block': L_heater_block,
    'Resolution': Resolution,
    'dx': dx,
    'L_total': L_total,
    'T_oil_setpoint': T_oil_setpoint,
    'T_ambient': T_ambient,
    'T_initial': T_initial,
    'OIL_PLACEMENT': OIL_PLACEMENT,
    'OIL_IS_FINITE': OIL_IS_FINITE,
    'Q_input_watts': Q_input_watts,
    'vol_heater': vol_heater,
    'Q_volumetric': Q_volumetric,
    'mat_ins': mat_ins,
    'mat_air_int': mat_air_int,
    'mat_oil': mat_oil,
    'mat_heat': mat_heat,
    'Time_Total': Time_Total,
    'dt': dt,
    'SNAPSHOT_INTERVAL_S': SNAPSHOT_INTERVAL_S,
    'Animation_Speedup': Animation_Speedup,
}

# Calculate n_steps for logging
n_steps = int(Time_Total / dt)

# Save experiment log before simulation starts
if SAVE_SNAPSHOTS:
    experiment_params['n_steps'] = n_steps
    save_experiment_log(SNAPSHOT_FOLDER, EXPERIMENT_DESCRIPTION, experiment_params)

# ==========================================
# 🚀 SIMULATION
# ==========================================

T = np.ones_like(X) * T_initial

# Set initial oil temp (whether finite or infinite, it starts at setpoint)
T[mask_oil] = T_oil_setpoint

# n_steps already calculated above for logging

# Animation Logic
TARGET_FPS = 30
real_duration = Time_Total / Animation_Speedup
total_anim_frames = int(real_duration * TARGET_FPS)
anim_save_interval = max(1, int(n_steps / total_anim_frames))

history_time, history_heater, history_oil = [], [], []
anim_frames = []

iterator = tqdm(range(n_steps), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [SimTime: {postfix}]', postfix="0s")

for n in iterator:
    # --- 1. Boundary Conditions ---
    
    # A. Oil BC
    if not OIL_IS_FINITE:
        # If infinite, force temp every step
        T[mask_oil] = T_oil_setpoint
    
    # B. Ambient Air BC
    # For single-sided mode, we need the "Ambient Air" to stay at room temp 
    # at the edges of the grid, or it will heat up infinitely too.
    # We apply Dirichlet BC to the outermost pixels of the Ambient region
    if OIL_PLACEMENT != "ALL":
        # Find edges of grid
        edge_mask = (X < dx) | (X > L_total-dx) | (Y < dx) | (Y > L_total-dx)
        # Apply Ambient Temp to edges that are Ambient Material
        T[mask_ambient & edge_mask] = T_ambient

    # --- 2. Physics Calculation ---
    d2T = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
           np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4 * T) / dx**2
    
    Q_term = np.zeros_like(T)
    Q_term[mask_heater] = Q_volumetric
    
    T += dt * (Alpha * d2T + Q_term / (Rho * Cp))
    
    # --- 3. Logging ---
    if n % 100 == 0:
        cur_time = n*dt
        t_h = np.mean(T[mask_heater])
        # Log Oil Temp (Simulated Mean)
        if np.sum(mask_oil) > 0:
            t_o = np.mean(T[mask_oil])
        else:
            t_o = T_oil_setpoint

        history_time.append(cur_time)
        history_heater.append(t_h)
        history_oil.append(t_o)
        
        if n % 500 == 0: 
            iterator.set_postfix_str(f"{int(cur_time)}s | Heat: {t_h-273.15:.0f}C | Oil: {t_o-273.15:.1f}C")

    # --- 4. Snapshots ---
    if SAVE_SNAPSHOTS and (n * dt) % SNAPSHOT_INTERVAL_S < dt:
        plt.ioff()
        fig_s, ax_s = plt.subplots(figsize=(6,6))
        im_s = ax_s.imshow(T, cmap='inferno', origin='lower', extent=[0, L_total*1000, 0, L_total*1000], vmin=T_initial, vmax=T_oil_setpoint+50)
        
        # Add colorbar
        cbar_s = plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)
        cbar_s.set_label('Temperature [K]', fontsize=10)
        
        # Draw Oil Boundary
        if np.sum(mask_oil) > 0:
            ax_s.contour(X*1000, Y*1000, mask_oil, levels=[0.5], colors='blue', linewidths=1.0)
        # Draw Box
        ax_s.contour(X*1000, Y*1000, mask_ins, levels=[0.5], colors='cyan', linewidths=0.5)
        
        current_time = n * dt
        ax_s.set_title(f"T={current_time:.0f}s | Mode: {OIL_PLACEMENT}")
        ax_s.set_xlabel("x [mm]")
        ax_s.set_ylabel("y [mm]")
        
        # Save with time-based filename instead of step number
        time_str = f"t{current_time:07.0f}s"  # Format: t0000120s for 120 seconds
        plt.savefig(f"{SNAPSHOT_FOLDER}/{time_str}.png", dpi=80, bbox_inches='tight')
        plt.close(fig_s)

    # --- 5. Animation Storage ---
    if n % anim_save_interval == 0:
        anim_frames.append(T.copy())

# Final Capture
anim_frames.append(T.copy())
print("\nSimulation Complete.")

# ==========================================
# 📈 1. TEMPERATURE EVOLUTION GRAPH (Save as PNG)
# ==========================================
if len(history_time) > 0:
    fig_temp = plt.figure(figsize=(10, 5))
    plt.plot(history_time, history_heater, 'r-', label='Heater', linewidth=2)
    plt.plot(history_time, history_oil, 'b--', label='Oil Avg', linewidth=2)
    plt.axhline(T_oil_setpoint, color='green', linestyle=':', label='Oil Setpoint', linewidth=1.5)
    plt.title(f"Temperature Evolution (Finite Oil: {OIL_IS_FINITE})", fontsize=14, fontweight='bold')
    plt.ylabel("Temperature [K]", fontsize=12)
    plt.xlabel("Time [s]", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save PNG to snapshot folder
    if SAVE_SNAPSHOTS:
        temp_plot_filename = os.path.join(SNAPSHOT_FOLDER, "temperature_evolution.png")
        plt.savefig(temp_plot_filename, dpi=150, bbox_inches='tight')
        print(f"Temperature evolution plot saved to: {temp_plot_filename}")
    
    plt.show()

# ==========================================
# 🎥 2. ANIMATION
# ==========================================
print("Preparing Animation...")
fig, ax = plt.subplots(figsize=(7, 7))
extent_mm = [0, L_total*1000, 0, L_total*1000]

# Initial Plot
im = ax.imshow(anim_frames[0], cmap='inferno', origin='lower', extent=extent_mm, vmin=T_initial, vmax=np.max(anim_frames[-1]))

# Contours to show where Oil is
if np.sum(mask_oil) > 0:
    ax.contour(X*1000, Y*1000, mask_oil, levels=[0.5], colors='blue', linewidths=1.5, linestyles='dashed')
    # Add text label for oil
    if OIL_PLACEMENT == "LEFT":
        ax.text(5, L_total*500, "OIL", color='blue', fontweight='bold')
    elif OIL_PLACEMENT == "ALL":
        ax.text(5, 5, "OIL", color='blue', fontweight='bold')

ax.contour(X*1000, Y*1000, mask_ins, levels=[0.5], colors='cyan', linewidths=1)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Temperature [K]')
title = ax.set_title("Replay")
ax.set_xlabel("mm"); ax.set_ylabel("mm")

def update(i):
    im.set_data(anim_frames[i])
    t_cur = i * anim_save_interval * dt
    title.set_text(f"Time: {t_cur:.0f}s")
    return [im, title]

ani = FuncAnimation(fig, update, frames=len(anim_frames), interval=1000/TARGET_FPS, blit=False)
plt.show()