import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available, using simple progress indicator")

# ==============================
# PARAMETER ZONE – ADJUST HERE
# ==============================

# Geometry
L_air = 0.086          # [m] air cavity width
t_ins = 0.004          # [m] insulation thickness (variable)
L_outside = 0.02       # [m] outside world visualization (20 mm)
L_total = L_air + 2*t_ins + 2*L_outside  # total domain width
Nx, Ny = 120, 120      # grid resolution (increased to show outside)

# Thermal properties
# options: 'aerogel', 'ptfe', or 'custom' - this does NOT consider another further layer. It also assumes worst case of the metal interface being 100% conductive, and neglegible thickness.
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

# Heat source - distributed over small area at center
Q_total = 8.0           # [W] total heat (distributed over small region)

# Boundary temperatures
T_ext = 273.15 + 270.0  # [K] oil bath at 270°C
T_init = 293.0          # [K] starting temp (20°C)
T_max_source = 1000.0   # [K] maximum temperature cap for heat source region

# Time stepping
dt = 0.05               # [s] (will be adjusted if needed for stability)
t_max = 10           # [s] - length of time to run the experiment. This will need to be extended ot the running time of the motor
save_every = 100         # plot every nth time step (will be adjusted)

# ======================================
# SETUP COMPUTATIONAL GRID AND PROPERTIES
# ======================================
dx = L_total / Nx

# Calculate stable time step: dt < dx²/(4*alpha_max) for 2D explicit scheme
alpha_air = k_air / (rho_air * cp_air)
alpha_ins = k_ins / (rho_ins * cp_ins)
alpha_max = max(alpha_air, alpha_ins)
dt_stable = 0.4 * dx**2 / (4 * alpha_max)  # Use 0.4 for safety margin
if dt_stable < dt:  # Only override if calculated dt is smaller
    dt = dt_stable
    print(f"Time step adjusted to {dt:.6f} s for numerical stability")
save_every = max(1, int(0.1 / dt))  # save roughly every 0.1 s

x = np.linspace(0, L_total, Nx)
y = np.linspace(0, L_total, Ny)
X, Y = np.meshgrid(x, y)

# Define material regions
# Outside world: 0 to L_outside and L_total-L_outside to L_total
# Insulation: L_outside to L_outside+t_ins and L_total-L_outside-t_ins to L_total-L_outside
# Air: L_outside+t_ins to L_total-L_outside-t_ins

mask_outside = ((X < L_outside) | (X > L_total - L_outside) | 
                (Y < L_outside) | (Y > L_total - L_outside))

mask_ins = ((X >= L_outside) & (X < L_outside + t_ins)) | \
           ((X > L_total - L_outside - t_ins) & (X <= L_total - L_outside)) | \
           ((Y >= L_outside) & (Y < L_outside + t_ins)) | \
           ((Y > L_total - L_outside - t_ins) & (Y <= L_total - L_outside))

mask_air = ~(mask_outside | mask_ins)

# Thermal diffusivity alpha = k/(rho*cp)
alpha = np.zeros_like(X)
alpha[mask_air] = k_air / (rho_air * cp_air)
alpha[mask_ins] = k_ins / (rho_ins * cp_ins)
alpha[mask_outside] = k_air / (rho_air * cp_air)  # outside also air-like

# Initial temperature field
# Air and insulation start at 20°C (293K)
# Outside world is at 270°C (543.15K) - constant heat source
T = np.full((Nx, Ny), T_init)  # Start with 293K everywhere
T[mask_air] = T_init  # Air region at 293K (20°C)
T[mask_ins] = T_init  # Insulation region at 293K (20°C)
T[mask_outside] = T_ext  # Outside world at 270°C - constant heat source

# ====================================
# HEAT SOURCE - DISTRIBUTED REGION
# ====================================
# Find grid point closest to center of air region (point source location)
cx = L_outside + t_ins + L_air/2  # center of air region in x
cy = L_outside + t_ins + L_air/2  # center of air region in y
# Find indices of center point
i_center = int(np.round(cx / dx))
j_center = int(np.round(cy / dx))
# Ensure it's within bounds
i_center = max(1, min(Nx-2, i_center))
j_center = max(1, min(Ny-2, j_center))

# Verify center point is in air region
if not mask_air[i_center, j_center]:
    print(f"Warning: Heat source at ({i_center}, {j_center}) is not in air region!")
    # Find nearest air point
    air_indices = np.where(mask_air)
    dists = np.sqrt((air_indices[0] - i_center)**2 + (air_indices[1] - j_center)**2)
    nearest_idx = np.argmin(dists)
    i_center = air_indices[0][nearest_idx]
    j_center = air_indices[1][nearest_idx]
    print(f"Adjusted to nearest air point: ({i_center}, {j_center})")

# Heat source: distribute 8W over a small area (not a single point to avoid unrealistic temps)
# Physical depth in z-direction (perpendicular to 2D plane)
space_length = 0.120 # [m] - 120 mm specified based on the RFQ
depth = space_length - (0.015 - 2*0.008 - 2*0.008) # [m] depth in z-direction: stack length + 2*winding_overhang + 2*insulation_thickness + end_frame_thickness
vol_per_cell = dx*dx*depth  # volume per cell [m³]

# Create a small heat source region (e.g., 3x3 or 5x5 cells around center)
# This prevents unrealistic temperature spikes from point sources
source_size_cells = 3  # Use 3x3 cell region for heat source 
""" 
Note that, although the heat source around the cell would suggest the electronics components around it are being cooked, we are really interested in the average temperature of the air inside. 
Ideally the 8 W heat source IS a singularity, but for it to be a heat source, it DOES still need a temperature, for the physics equations to be resolved ocrrectly.
"""
source_radius_cells = source_size_cells // 2

# Find all cells in the heat source region
Q_map = np.zeros_like(T)
source_cells = 0
for di in range(-source_radius_cells, source_radius_cells + 1):
    for dj in range(-source_radius_cells, source_radius_cells + 1):
        i_src = i_center + di
        j_src = j_center + dj
        # Only add if within bounds and in air region
        if (0 < i_src < Nx-1 and 0 < j_src < Ny-1 and 
            mask_air[i_src, j_src]):
            Q_map[i_src, j_src] = 1.0
            source_cells += 1

# Distribute power evenly over all source cells
if source_cells > 0:
    Q_per_cell = Q_total / source_cells  # [W] per cell
    Q_rate_per_cell = Q_per_cell / (rho_air * cp_air * vol_per_cell)  # [K/s] per cell
    Q_map = Q_map * Q_rate_per_cell
    print(f"Heat source: {Q_total} W distributed over {source_cells} cells ({source_size_cells}×{source_size_cells} region)")
    print(f"Power per cell: {Q_per_cell:.4f} W")
    print(f"Temperature rate per cell: {Q_rate_per_cell:.4f} K/s")
else:
    print("Warning: No valid heat source cells found!")
    Q_map = np.zeros_like(T)

# Calculate volumes
n_air_cells = np.sum(mask_air)
n_ins_cells = np.sum(mask_ins)
total_air_volume = n_air_cells * vol_per_cell
total_ins_volume = n_ins_cells * vol_per_cell

# Expected volumes (geometric calculation)
expected_air_volume = L_air * L_air * depth
# Insulation volume: outer box minus inner air box
L_outer = L_air + 2*t_ins
expected_ins_volume = (L_outer * L_outer - L_air * L_air) * depth

print(f"\n=== VOLUME CALCULATION ===")
print(f"Grid spacing dx: {dx*1000:.3f} mm")
print(f"Cell dimensions: {dx*1000:.3f} mm × {dx*1000:.3f} mm × {depth*1000:.1f} mm")
print(f"\nVolume per cell: {vol_per_cell*1e9:.2f} mm³")
print(f"\nAir region:")
print(f"  Number of air cells: {n_air_cells}")
print(f"  Total air volume: {total_air_volume*1e9:.2f} mm³ ({total_air_volume*1e-6:.4f} L)")
print(f"  Expected (geometric): {expected_air_volume*1e9:.2f} mm³ ({expected_air_volume*1e-6:.4f} L)")
print(f"\nInsulation region:")
print(f"  Number of insulation cells: {n_ins_cells}")
print(f"  Total insulation volume: {total_ins_volume*1e9:.2f} mm³ ({total_ins_volume*1e-6:.4f} L)")
print(f"  Expected (geometric): {expected_ins_volume*1e9:.2f} mm³ ({expected_ins_volume*1e-6:.4f} L)") 

# ====================================
# TIME INTEGRATION LOOP
# ====================================
n_steps = int(t_max / dt)
frames = []  # store temperature fields
time_points = []
inner_temps = []
source_temps = []  # track temperature at heat source location

print(f"\n=== SIMULATION ===")
print(f"Total time steps: {n_steps:,}")
print(f"Simulation time: {t_max:.1f} s ({t_max/3600:.2f} hours)")
print(f"Time step: {dt:.6f} s")
print(f"Saving every {save_every} steps (~{save_every*dt:.1f} s)")
print(f"Expected frames: ~{n_steps//save_every:,}\n")

# Create progress bar
if HAS_TQDM:
    iterator = tqdm(range(n_steps), desc="Simulating", unit="step", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
else:
    # Simple progress indicator
    progress_interval = max(1, n_steps // 100)  # Update every 1%
    iterator = range(n_steps)

for n in iterator:
    Tn = T.copy()
    # finite difference, internal nodes
    T[1:-1,1:-1] = Tn[1:-1,1:-1] + (
        alpha[1:-1,1:-1]*dt/dx**2 * (
            Tn[2:,1:-1] + Tn[:-2,1:-1] + Tn[1:-1,2:] + Tn[1:-1,:-2] - 4*Tn[1:-1,1:-1])
        + Q_map[1:-1,1:-1]*dt  # Distributed heat source
    )
    
    # Cap heat source temperature to prevent unrealistic values
    source_mask = Q_map > 0
    T[source_mask] = np.minimum(T[source_mask], T_max_source)
    
    # Boundary condition: constant T_ext at outer boundary (infinite heat source/sink at 270°C)
    T[0,:] = T_ext
    T[-1,:] = T_ext
    T[:,0] = T_ext
    T[:,-1] = T_ext
    
    # Keep outside world at constant temperature (infinite heat source/sink)
    T[mask_outside] = T_ext
    
    # Track temperatures
    inner_temps.append(np.mean(T[mask_air]))
    # Track temperature AROUND the heat source (not inside it) - average of adjacent cells
    source_mask = Q_map > 0
    nearby_temp = []
    # Sample cells around the source region (not in it)
    for di in range(-source_size_cells, source_size_cells + 3):
        for dj in range(-source_size_cells, source_size_cells + 3):
            i_near = i_center + di
            j_near = j_center + dj
            # Only count cells that are in air, not in source, and adjacent to source
            if (0 <= i_near < Nx and 0 <= j_near < Ny and 
                mask_air[i_near, j_near] and Q_map[i_near, j_near] == 0):
                # Check if adjacent to source region
                is_adjacent = False
                for di2 in [-1, 0, 1]:
                    for dj2 in [-1, 0, 1]:
                        if (0 <= i_near + di2 < Nx and 0 <= j_near + dj2 < Ny and
                            Q_map[i_near + di2, j_near + dj2] > 0):
                            is_adjacent = True
                            break
                    if is_adjacent:
                        break
                if is_adjacent:
                    nearby_temp.append(T[i_near, j_near])
    if nearby_temp:
        source_temps.append(np.mean(nearby_temp))  # Average temp around source
    else:
        source_temps.append(T[i_center, j_center])  # fallback
    time_points.append(n*dt)
    
    # Simple progress indicator (if tqdm not available)
    if not HAS_TQDM and n % progress_interval == 0:
        progress = 100.0 * n / n_steps
        elapsed_time = n * dt
        print(f"Progress: {progress:.1f}% | Time: {elapsed_time:.1f} s / {t_max:.1f} s | "
              f"Avg air temp: {inner_temps[-1]-273.15:.1f}°C", end='\r')
    
    # Save for visualization
    if n % save_every == 0:
        frames.append(T.copy())

# Simulation complete
if not HAS_TQDM:
    print()  # New line after progress indicator
print(f"\nSimulation complete! Generated {len(frames)} frames.")
print(f"Final average air temperature: {inner_temps[-1]-273.15:.2f}°C")
print(f"Final temperature around heat source: {source_temps[-1]-273.15:.2f}°C\n")

# ====================================
# PLOTTING RESULTS
# ====================================
fig, ax = plt.subplots(figsize=(10, 10))
# Set colorbar limits to show relevant temperature range
# Focus on range from initial temp to oil bath temp + small margin
# This allows us to see differences at interfaces clearly
T_plot_min = T_init  # Start at 20°C (293K)
T_plot_max = T_ext + 50.0  # Go slightly above oil bath (270°C = 543K, so up to ~593K)
im = ax.imshow(frames[0], extent=[0,L_total*1000,0,L_total*1000], origin='lower', 
               cmap='inferno', vmin=T_plot_min, vmax=T_plot_max)
plt.title("Temperature field evolution")
cbar = plt.colorbar(im, label='Temperature [K]', ax=ax)
# Add temperature in Celsius on secondary axis
cbar.ax.set_ylabel('Temperature [K]', rotation=270, labelpad=20)
# Add tick marks for key temperatures
ticks_k = np.array([293, 350, 400, 450, 500, 543, 593])  # Key temps in K
cbar.set_ticks(ticks_k)
tick_labels = [f'{T:.0f}K\n({T-273.15:.0f}°C)' for T in ticks_k]
cbar.set_ticklabels(tick_labels)

# Draw boundaries to show layers
# Air-insulation boundary
ax.axvline(x=(L_outside+t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Air/Insulation')
ax.axvline(x=(L_total-L_outside-t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=(L_outside+t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=(L_total-L_outside-t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
# Insulation-outside boundary
ax.axvline(x=L_outside*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='Insulation/Outside')
ax.axvline(x=(L_total-L_outside)*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=L_outside*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=(L_total-L_outside)*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
# Mark heat source
ax.plot(cx*1000, cy*1000, 'r*', markersize=15, label='Heat Source')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.legend(loc='upper right')

def animate(i):
    im.set_data(frames[i])
    ax.set_title(f"Temperature field at t = {i*dt*save_every:.1f} s")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=150)
plt.show()

# Plot temperature evolution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time_points, inner_temps, 'b-', label='Average Air Temperature', linewidth=2)
plt.plot(time_points, source_temps, 'r-', label='Temperature Around Heat Source', linewidth=2)
plt.axhline(y=T_ext, color='g', linestyle='--', label=f'Boundary ({T_ext:.1f} K = {T_ext-273.15:.1f}°C)')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.title('Temperature Evolution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_points, np.array(inner_temps)-273.15, 'b-', label='Average Air Temperature', linewidth=2)
plt.plot(time_points, np.array(source_temps)-273.15, 'r-', label='Temperature Around Heat Source', linewidth=2)
plt.axhline(y=T_ext-273.15, color='g', linestyle='--', label=f'Boundary ({T_ext-273.15:.1f}°C)')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature Evolution (Celsius)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print results
print(f"\n=== RESULTS ===")
print(f"Geometry:")
print(f"  Air cavity: {L_air*1000:.1f} mm × {L_air*1000:.1f} mm")
print(f"  Insulation thickness: {t_ins*1000:.1f} mm")
print(f"  Outside world visualization: {L_outside*1000:.1f} mm")
print(f"Heat source location: center at ({cx*1000:.1f} mm, {cy*1000:.1f} mm), distributed over {source_cells} cells")
print(f"Initial air temperature: {T_init:.1f} K ({T_init-273.15:.1f}°C)")
print(f"Boundary/Outside temperature: {T_ext:.1f} K ({T_ext-273.15:.1f}°C) - constant")
print(f"Final temperature around heat source: {source_temps[-1]:.2f} K ({source_temps[-1]-273.15:.2f}°C)")
print(f"Final average air temperature: {inner_temps[-1]:.2f} K ({inner_temps[-1]-273.15:.2f}°C)")
print(f"Maximum temperature around heat source reached: {max(source_temps):.2f} K ({max(source_temps)-273.15:.2f}°C)")