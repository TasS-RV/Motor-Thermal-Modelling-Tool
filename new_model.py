import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm # Imported directly as requested

# ==========================================
# 🎛️ CONTROL PANEL
# ==========================================

# --- Snapshot Settings ---
SAVE_SNAPSHOTS      = True           # Set to False to disable saving to disk
SNAPSHOT_INTERVAL_S = 60.0          # Save an image every 120 simulation seconds
SNAPSHOT_FOLDER     = "sim_snapshots_v3" # Folder name

# --- Geometry [m] ---
L_air_inner    = 0.094       
t_insulation   = 0.004       
t_oil_margin   = 0.050       
D_depth        = 0.080       

# --- Heater Source ---
L_heater_block = 0.030      
Q_input_watts  = 8.0        

# --- Temperatures [K] ---
T_oil_bath     = 273.15 + 170.0  # 170°C
T_initial      = 293.15          # 20°C

# --- Simulation Settings ---
Resolution     = 140         # Grid resolution
Time_Total     = 180      # Total simulation time in seconds (e.g. 30 mins)
Animation_Speedup = 50.0     # For the playback window speed

ins_type = 'aerogel'   

# Select insulation material properties
if ins_type.lower() == 'aerogel':
    k_ins = 0.02        # W/mK - conductivity from: https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html
    rho_ins = 150       # kg/m3 - https://www.mdpi.com/2073-4360/14/7/1456?. Various ranges suggest 30-350 kg/m^3. We are using a rough intermediate value. Alternative from actual supplier: https://www.keepinsulation.com/aerogel/aerogel-felt/silica-aerogel-thermal-insulation-roll.html
    cp_ins = 1000       # J/kgK - for aerogel this is sensible, perhaps slightly different. Given it is very low density, wiht high proportion of air (>95%).
elif ins_type.lower() == 'ptfe':
    k_ins = 0.25  # See below!
    rho_ins = 2200 # Density and Thermal conducitivity from: https://en.wikipedia.org/wiki/Polytetrafluoroethylene
    cp_ins = 1010 # Based on the Specific heat at 23C from https://adtech.co.uk/application/files/8516/0500/0920/Adtech_PTFE_General_Properties_2020.pdf
else:
    k_ins = 0.1
    rho_ins = 500
    cp_ins = 1000

# Air properties
k_air = 0.0262          # W/mK
rho_air = 1.225         # kg/m3
cp_air = 1005           # J/kgK

# --- Materials ---
mat_ins  = {'k': k_ins,  'rho': rho_ins,  'cp': cp_ins} # Parameters ti be set based on the selected insulator
mat_air  = {'k': k_air, 'rho': rho_air,  'cp': cp_air} # Air
mat_heat = {'k': 160.0, 'rho': 2700.0, 'cp': 900.0}  # Heater Core
mat_oil  = {'k': 0.15,  'rho': 800.0,  'cp': 2000.0} # Oil

# ==========================================
# ⚙️ SETUP
# ==========================================

# 1. Create Snapshot Folder
if SAVE_SNAPSHOTS:
    if not os.path.exists(SNAPSHOT_FOLDER):
        os.makedirs(SNAPSHOT_FOLDER)
        print(f"Created folder: {SNAPSHOT_FOLDER}")

# 2. Geometry Setup
L_total = L_air_inner + 2*t_insulation + 2*t_oil_margin
dx = L_total / Resolution
dy = dx
x = np.linspace(0, L_total, Resolution)
y = np.linspace(0, L_total, Resolution)
X, Y = np.meshgrid(x, y)

# 3. Define Regions (Masks)
center = L_total / 2
dist_x = np.abs(X - center)
dist_y = np.abs(Y - center)

mask_heater   = (dist_x < L_heater_block/2) & (dist_y < L_heater_block/2)
limit_air     = L_air_inner / 2
mask_air_zone = (dist_x < limit_air) & (dist_y < limit_air)
mask_air      = mask_air_zone & (~mask_heater)
limit_ins     = limit_air + t_insulation
mask_ins_zone = (dist_x < limit_ins) & (dist_y < limit_ins)
mask_ins      = mask_ins_zone & (~mask_air_zone)
mask_oil      = ~mask_ins_zone

# 4. Map Material Properties
K   = np.zeros_like(X); Rho = np.zeros_like(X); Cp  = np.zeros_like(X)

def apply_mat(mask, mat):
    K[mask] = mat['k']; Rho[mask] = mat['rho']; Cp[mask] = mat['cp']

apply_mat(mask_oil, mat_oil)
apply_mat(mask_ins, mat_ins)
apply_mat(mask_air, mat_air)
apply_mat(mask_heater, mat_heat)

# 5. Physics Prep
vol_heater = np.sum(mask_heater) * dx * dy * D_depth
Q_volumetric = Q_input_watts / vol_heater

Alpha = K / (Rho * Cp)
alpha_max = np.max(Alpha[~mask_oil]) 
dt = (0.8 * dx**2) / (4 * alpha_max)

print(f"Time Step: {dt:.5f} s")
print(f"Simulation Duration: {Time_Total} s")
print(f"Saving Snapshots every {SNAPSHOT_INTERVAL_S} s")

# ==========================================
# 🚀 RUN SIMULATION
# ==========================================

T = np.ones_like(X) * T_initial
T[mask_oil] = T_oil_bath

n_steps = int(Time_Total / dt)
steps_per_snapshot = int(SNAPSHOT_INTERVAL_S / dt)

# Data Logging
history_time = []
history_heater = []
history_air = []

# Animation Frame Storage
anim_frames = []
anim_save_interval = int(n_steps / (Time_Total/Animation_Speedup * 30)) 
if anim_save_interval < 1: anim_save_interval = 1

# Setup Progress Bar directly
iterator = tqdm(range(n_steps), 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Time: {postfix}]',
                postfix="0.0s | T_heat: 20C")

for n in iterator:
    # 1. Physics
    T[mask_oil] = T_oil_bath
    d2T = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
           np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4 * T) / dx**2
    Q_term = np.zeros_like(T)
    Q_term[mask_heater] = Q_volumetric
    T += dt * (Alpha * d2T + Q_term / (Rho * Cp))
    
    # 2. Data Logging (Every 100 steps to keep arrays manageable)
    if n % 100 == 0:
        current_time = n * dt
        t_h_mean = np.mean(T[mask_heater])
        t_a_mean = np.mean(T[mask_air])
        
        history_time.append(current_time)
        history_heater.append(t_h_mean)
        history_air.append(t_a_mean)
        
        # Update Progress Bar Text (every 500 steps to avoid flicker)
        if n % 500 == 0:
            iterator.set_postfix_str(f"{current_time:.1f}s | T_heat: {t_h_mean-273.15:.1f}C")

    # 3. Save Snapshot to Disk
    if SAVE_SNAPSHOTS and (n % steps_per_snapshot == 0):
        # Only check this condition so we don't slow down the main loop constantly
        sim_time = n * dt
        plt.ioff()
        fig_snap, ax_snap = plt.subplots(figsize=(8, 8))
        im = ax_snap.imshow(T, cmap='inferno', origin='lower', 
                           extent=[0, L_total*1000, 0, L_total*1000],
                           vmin=T_initial, vmax=T_oil_bath+10)
        ax_snap.contour(X*1000, Y*1000, mask_ins, levels=[0.5], colors='cyan', linewidths=0.5)
        ax_snap.set_title(f"Time: {sim_time:.1f} s | Heater: {np.mean(T[mask_heater])-273.15:.1f}°C")
        plt.colorbar(im, ax=ax_snap, label='Temp [K]')
        filename = f"{SNAPSHOT_FOLDER}/step_{n:06d}_time_{int(sim_time)}s.png"
        plt.savefig(filename, dpi=80)
        plt.close(fig_snap)
        
    # 4. Store frame for Animation
    if n % anim_save_interval == 0:
        anim_frames.append(T.copy())

# Capture final state
anim_frames.append(T.copy())
history_time.append(n*dt)
history_heater.append(np.mean(T[mask_heater]))
history_air.append(np.mean(T[mask_air]))

print(f"\nSimulation Complete.")

# ==========================================
# 📈 FINAL ANALYSIS GRAPHS
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(history_time, history_heater, label='Heater Block', color='red', linewidth=2)
plt.plot(history_time, history_air, label='Avg Air Temp', color='orange', linewidth=2)
plt.axhline(T_oil_bath, color='blue', linestyle='--', label='Oil Boundary (170°C)')
plt.xlabel('Simulation Time [s]')
plt.ylabel('Temperature [K]')
plt.title(f'Temperature Evolution (Total Time: {Time_Total}s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 🎥 PLAYBACK ANIMATION
# ==========================================
print("Launching Animation Window...")
fig, ax = plt.subplots(figsize=(8, 8))
extent_mm = [0, L_total*1000, 0, L_total*1000]
im = ax.imshow(anim_frames[0], cmap='inferno', origin='lower', extent=extent_mm,
               vmin=T_initial, vmax=T_oil_bath + 20)

ax.contour(X*1000, Y*1000, mask_ins, levels=[0.5], colors='cyan', linewidths=1)
ax.set_xlabel("mm"); ax.set_ylabel("mm")
title = ax.set_title("Replay")

def update(i):
    im.set_data(anim_frames[i])
    # Calculate approx time for frame
    t_approx = i * anim_save_interval * dt
    title.set_text(f"Playback: {t_approx:.1f}s")
    return [im, title]

ani = FuncAnimation(fig, update, frames=len(anim_frames), interval=30, blit=False)
plt.show()