import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import tqdm 
# ==========================================
# 🎛️ CONTROL PANEL
# ==========================================

# --- Geometry [m] ---
# Inner air cavity size (94mm)
L_air_inner   = 0.094       
# Insulation thickness (4mm)
t_insulation  = 0.004       
# Oil Bath Margin (Increased to 50mm to look like a bath)
t_oil_margin  = 0.050       
# Depth of the unit (80mm) - used for Volume calculation
D_depth       = 0.080       

# --- Heater Source ---
# We use a 30x30mm block to distribute the 8W (prevents singularity)
L_heater_block = 0.030      
Q_input_watts  = 8.0        

# --- Temperatures [K] ---
T_oil_bath    = 273.15 + 170.0  # 170°C (Fixed Boundary)
T_initial     = 293.15          # 20°C (Start Temp)

# --- Simulation Settings ---
Resolution    = 140         # Grid resolution (140x140)
Time_Total    = 600        # Total simulation duration in seconds
Animation_Speedup = 50.0    # Playback speed (e.g., 50x real time)

# --- Materials ---
# Aerogel-like insulation
mat_ins  = {'k': 0.02,  'rho': 150.0,  'cp': 1000.0}
# Air
mat_air  = {'k': 0.026, 'rho': 1.225,  'cp': 1005.0}
# Heater Core (Aluminum/PCB mix thermal mass)
mat_heat = {'k': 160.0, 'rho': 2700.0, 'cp': 900.0}
# Oil (Properties don't impact math much due to fixed temp, but good for completeness)
mat_oil  = {'k': 0.15,  'rho': 800.0,  'cp': 2000.0}

# ==========================================
# ⚙️ SETUP & PHYSICS
# ==========================================

# 1. Geometry Setup
# Total Width = Oil + Ins + Air + Ins + Oil
L_total = L_air_inner + 2*t_insulation + 2*t_oil_margin
dx = L_total / Resolution
dy = dx

# Create Coordinate Grid
x = np.linspace(0, L_total, Resolution)
y = np.linspace(0, L_total, Resolution)
X, Y = np.meshgrid(x, y)

# 2. Define Regions (Masks)
center = L_total / 2
dist_x = np.abs(X - center)
dist_y = np.abs(Y - center)

# Heater Mask (Center)
mask_heater = (dist_x < L_heater_block/2) & (dist_y < L_heater_block/2)

# Air Mask (Inner Box - Heater)
lim_air = L_air_inner / 2
mask_air_zone = (dist_x < lim_air) & (dist_y < lim_air)
mask_air = mask_air_zone & (~mask_heater)

# Insulation Mask (Outer Box - Air Zone)
lim_ins = lim_air + t_insulation
mask_ins_zone = (dist_x < lim_ins) & (dist_y < lim_ins)
mask_ins = mask_ins_zone & (~mask_air_zone)

# Oil Mask (Everything outside insulation)
mask_oil = ~mask_ins_zone

# 3. Map Material Properties to Grid
K   = np.zeros_like(X)
Rho = np.zeros_like(X)
Cp  = np.zeros_like(X)

# Helper function to apply properties
def apply_mat(mask, mat):
    K[mask]   = mat['k']
    Rho[mask] = mat['rho']
    Cp[mask]  = mat['cp']

apply_mat(mask_oil, mat_oil)     # Background
apply_mat(mask_ins, mat_ins)     # Insulation
apply_mat(mask_air, mat_air)     # Air
apply_mat(mask_heater, mat_heat) # Heater

# 4. Heat Source Calculation
# Calculate volume of the heater block in m^3
vol_heater = np.sum(mask_heater) * dx * dy * D_depth
Q_volumetric = Q_input_watts / vol_heater  # Watts / m^3

# 5. Stability Calculation
Alpha = K / (Rho * Cp)
alpha_max = np.max(Alpha[~mask_oil]) # Exclude oil boundary from stability check
dt = (0.8 * dx**2) / (4 * alpha_max) # 0.8 safety factor

print(f"--- SIMULATION READY ---")
print(f"Total Width: {L_total*1000:.1f} mm")
print(f"Oil Margin:  {t_oil_margin*1000:.1f} mm (per side)")
print(f"Heater Vol:  {vol_heater*1e6:.1f} cm3")
print(f"Time Step:   {dt:.5f} s")

# ==========================================
# 🚀 RUN SIMULATION
# ==========================================

# Initial Condition
T = np.ones_like(X) * T_initial
T[mask_oil] = T_oil_bath # Oil starts and stays at 170C

n_steps = int(Time_Total / dt)

# Animation Settings
target_fps = 30
# We want the animation to last: Time_Total / Speedup
anim_duration = Time_Total / Animation_Speedup
total_frames_needed = int(anim_duration * target_fps)
save_every_n_steps = int(n_steps / total_frames_needed)
if save_every_n_steps < 1: save_every_n_steps = 1

frames = [] # To store snapshots

print(f"Simulating {Time_Total}s... (Saving {total_frames_needed} frames)")

iterator = range(n_steps)
if HAS_TQDM:
    iterator = tqdm(range(n_steps), unit="step")

for n in iterator:
    # 1. Enforce Boundary (Infinite Oil Bath)
    T[mask_oil] = T_oil_bath
    
    # 2. Compute Laplacian (Finite Difference)
    d2T = (
        np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) -
        4 * T
    ) / dx**2
    
    # 3. Add Heat Source
    Q_term = np.zeros_like(T)
    Q_term[mask_heater] = Q_volumetric
    
    # 4. Update Temperature Field
    T += dt * (Alpha * d2T + Q_term / (Rho * Cp))
    
    # 5. Save Frame
    if n % save_every_n_steps == 0:
        frames.append(T.copy())

# Ensure last frame is caught
frames.append(T.copy())

# ==========================================
# 🎥 ANIMATION GENERATION
# ==========================================

print("Generating Animation...")

fig, ax = plt.subplots(figsize=(8, 8))

# Setup the plot extent to be in mm
extent_mm = [0, L_total*1000, 0, L_total*1000]

# Initial Plot
# We use vmin/vmax to keep the color scale fixed throughout the animation
im = ax.imshow(frames[0], cmap='inferno', origin='lower', extent=extent_mm,
               vmin=T_initial, vmax=T_oil_bath + 20)

# Add visual overlays for the box structure
# Convert mask boundaries to contours for clean lines
ax.contour(X*1000, Y*1000, mask_ins, levels=[0.5], colors='cyan', linewidths=1, linestyles='solid')
ax.contour(X*1000, Y*1000, mask_heater, levels=[0.5], colors='white', linewidths=1, linestyles='dashed')

# Formatting
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Temperature [K]')
ax.set_title("Initializing...")
ax.set_xlabel("Width [mm]")
ax.set_ylabel("Height [mm]")

# Update function for animation
def update(frame_idx):
    # Update image data
    im.set_data(frames[frame_idx])
    
    # Calculate simulation time for title
    sim_time = frame_idx * save_every_n_steps * dt
    
    # Get current center temp
    center_temp = np.mean(frames[frame_idx][mask_heater])
    air_temp = np.mean(frames[frame_idx][mask_air])
    
    ax.set_title(f"Time: {sim_time:.1f}s | Heater: {center_temp:.0f}K | Air: {air_temp:.0f}K")
    return [im]

# Create Animation
ani = FuncAnimation(fig, update, frames=len(frames), interval=1000/target_fps, blit=False)

print("Displaying animation window...")
plt.show()

# Optional: Save logic (commented out)
# ani.save('simulation_bath.gif', writer='pillow', fps=target_fps)