import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
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
Nx, Ny = 400, 400      # grid resolution (increased to show outside)

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
enable_internal_heat_source = True  # Set to False to disable internal heat source and validate boundary heating alone
Q_total = 8.0           # [W] total heat (distributed over small region) - set to 0 if enable_internal_heat_source is False

# Boundary temperatures - all temperatures are set in kelvin
T_ext = 273.15 + 170.0  # [K] oil bath at 170°C
T_init = 293.0          # [K] starting temp (20°C)
T_max_source = 273.15 + 170.0   # [K] maximum temperature cap for heat source region

# Time stepping
dt = 0.2               # [s] (will be adjusted if needed for stability)
t_max = 1200          # [s] - length of time to run the experiment. This will need to be extended ot the running time of the motor
save_period = 10.0      # [s] time period between saved frames (save_every will be automatically calculated based on actual dt)

# Image saving
num_snapshots = 20      # number of snapshot images to save (evenly distributed over total time)
save_snapshots = True   # set to False to disable snapshot saving
snapshot_folder = "Fast exp 2, local heat source - 3x3 cells"  # folder name to save snapshots

# Animation speed
playback_speed_ratio = 60.0  # e.g., 60 means 1 hour of simulation plays in 60 seconds (60x speed)

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

# Calculate save_every based on desired time period and actual time step
save_every = max(1, int(save_period / dt))  # save every save_period seconds
actual_save_period = save_every * dt  # actual time period achieved (may be slightly different due to integer rounding)
print(f"Frame saving: every {save_every} time steps = {actual_save_period:.3f} s (requested: {save_period:.3f} s)")

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



"""
ANCHOR: modify heat source parameters here!
"""
# Heat source: distribute 8W over a small area (not a single point to avoid unrealistic temps)
# Physical depth in z-direction (perpendicular to 2D plane)
space_length = 0.120 # [m] - 120 mm specified based on the RFQ
depth = space_length - (0.015 - 2*0.008 - 2*0.008) # [m] depth in z-direction: stack length + 2*winding_overhang + 2*insulation_thickness + end_frame_thickness
vol_per_cell = dx*dx*depth  # volume per cell [m³]

# Create a small heat source region (e.g., 3x3 or 5x5 cells around center)
# This prevents unrealistic temperature spikes from point sources
source_size_cells = 4  # Use 40x40 cell region for heat source 
""" 
Note that, although the heat source around the cell would suggest the electronics components around it are being cooked, we are really interested in the average temperature of the air inside. 
Ideally the 8 W heat source IS a singularity, but for it to be a heat source, it DOES still need a temperature, for the physics equations to be resolved ocrrectly.

When the source is too small, then the local heating rate is too high causing ridiculous/ cell temperature rise computations, but this is wrong.
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

# Calculate expected boundary heat flux for comparison (always calculate for physics check)
area_per_side = L_air * depth  # area of one side of air cavity
total_insulation_area = 4 * area_per_side + 2 * L_air * L_air  # 4 vertical + 2 horizontal sides
heat_flux_through_insulation = k_ins * (T_ext - T_init) / t_ins  # W/m²
estimated_boundary_heat = heat_flux_through_insulation * total_insulation_area  # W

# Print physics check
print(f"\n=== HEAT SOURCE PHYSICS CHECK ===")
print(f"Internal heat source enabled: {enable_internal_heat_source}")
if enable_internal_heat_source:
    print(f"Internal heat source: {Q_total} W")
else:
    print(f"Internal heat source: DISABLED (0 W) - validating boundary heating alone")
    Q_total = 0.0  # Override to zero if disabled

print(f"\nBoundary heating:")
print(f"  Boundary temp: {T_ext:.1f} K ({T_ext-273.15:.1f}°C)")
print(f"  Initial temp: {T_init:.1f} K ({T_init-273.15:.1f}°C)")
print(f"  Temperature difference: {T_ext - T_init:.1f} K")
print(f"  Insulation area: {total_insulation_area*1e4:.2f} cm²")
print(f"  Heat flux: {heat_flux_through_insulation:.1f} W/m²")
print(f"  Estimated boundary heat transfer: {estimated_boundary_heat:.2f} W")

# Distribute power evenly over all source cells (if enabled)
if enable_internal_heat_source and source_cells > 0 and Q_total > 0:
    # IMPORTANT: Q_total is the TOTAL power distributed over ALL source cells, not per cell!
    Q_per_cell = Q_total / source_cells  # [W] per cell = total power / number of cells
    Q_rate_per_cell = Q_per_cell / (rho_air * cp_air * vol_per_cell)  # [K/s] per cell
    
    # Verify total power: Q_per_cell * source_cells should equal Q_total
    total_power_check = Q_per_cell * source_cells
    
    Q_map = Q_map * Q_rate_per_cell  # Multiply Q_map (1.0 in source cells) by the heating rate
    
    print(f"\nInternal heat source details:")
    print(f"  Source region: {source_size_cells}×{source_size_cells} cells = {source_size_cells*source_size_cells} max cells")
    print(f"  Valid source cells (in air region): {source_cells} cells")
    print(f"  TOTAL power: {Q_total:.4f} W (distributed over ALL {source_cells} cells)")
    print(f"  Power per cell: {Q_total:.4f} W / {source_cells} cells = {Q_per_cell:.6f} W/cell")
    print(f"  Verification: {Q_per_cell:.6f} W/cell × {source_cells} cells = {total_power_check:.4f} W ✓")
    print(f"  Cell volume: {vol_per_cell*1e9:.2f} mm³")
    print(f"  Temperature rate per cell (if isolated): {Q_rate_per_cell:.4f} K/s")
    print(f"  → Larger source region ({source_size_cells}×{source_size_cells}) means GENTLER heating (less power per cell)")
    print(f"\n  Boundary heat / Internal heat = {estimated_boundary_heat/Q_total:.1f}x")
    print(f"  → The {Q_total}W internal source should be MINOR compared to boundary heating!")
    print(f"  → Final temperature should be dominated by boundary temp (~{T_ext-273.15:.0f}°C)")
    print(f"  → The internal source will only create a small local temperature rise")
elif enable_internal_heat_source:
    print("\nWarning: No valid heat source cells found! Internal heat source disabled.")
    Q_map = np.zeros_like(T)
else:
    # Internal heat source disabled - zero out Q_map
    Q_map = np.zeros_like(T)
    print(f"\n  Internal heat source is DISABLED")
    print(f"  → Temperature should be dominated entirely by boundary heating")
    print(f"  → Final temperature should approach boundary temp ({T_ext-273.15:.0f}°C)")

print()  # Empty line for readability

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
# PLOTTING PARAMETERS (defined early for snapshot saving)
# ====================================
T_plot_min = T_init  # Start at 20°C (293K)
T_plot_max = T_ext + 50.0  # Go slightly above oil bath (270°C = 543K, so up to ~593K)
ticks_k = np.array([293, 350, 400, 450, 500, 543, 593])  # Key temps in K for colorbar
tick_labels = [f'{T:.0f}K\n({T-273.15:.0f}°C)' for T in ticks_k]

# ====================================
# TIME INTEGRATION LOOP
# ====================================
n_steps = int(t_max / dt)
frames = []  # store temperature fields
time_points = []
inner_temps = []
source_temps = []  # track temperature at heat source location

# Calculate snapshot save times (evenly distributed)
snapshot_times = []
snapshot_save_indices = []
if save_snapshots and num_snapshots > 0:
    snapshot_interval = t_max / num_snapshots if num_snapshots > 1 else t_max
    for i in range(num_snapshots):
        t_snapshot = (i + 1) * snapshot_interval
        if t_snapshot <= t_max:
            snapshot_times.append(t_snapshot)
            snapshot_save_indices.append(int(t_snapshot / dt))
    # Create snapshot folder
    os.makedirs(snapshot_folder, exist_ok=True)
    print(f"Will save {len(snapshot_times)} snapshots at times: {[f'{t:.1f}s' for t in snapshot_times]}")

print(f"\n=== SIMULATION ===")
print(f"Total time steps: {n_steps:,}")
print(f"Simulation time: {t_max:.1f} s ({t_max/3600:.2f} hours)")
print(f"Time step: {dt:.6f} s")
print(f"Saving every {save_every} steps = every {actual_save_period:.3f} s (requested: every {save_period:.3f} s)")
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
    
    # Save snapshots at specified times
    if save_snapshots and n in snapshot_save_indices:
        snapshot_idx = snapshot_save_indices.index(n)
        t_snapshot = snapshot_times[snapshot_idx]
        
        # Create snapshot figure
        fig_snap, ax_snap = plt.subplots(figsize=(10, 10))
        im_snap = ax_snap.imshow(T, extent=[0, L_total*1000, 0, L_total*1000], origin='lower',
                                 cmap='inferno', vmin=T_plot_min, vmax=T_plot_max)
        ax_snap.set_title(f"Temperature field at t = {t_snapshot:.1f} s ({t_snapshot/60:.1f} min)")
        
        # Draw boundaries
        ax_snap.axvline(x=(L_outside+t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axvline(x=(L_total-L_outside-t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axhline(y=(L_outside+t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axhline(y=(L_total-L_outside-t_ins)*1000, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axvline(x=L_outside*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axvline(x=(L_total-L_outside)*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axhline(y=L_outside*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.axhline(y=(L_total-L_outside)*1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        ax_snap.plot(cx*1000, cy*1000, 'r*', markersize=15)
        
        cbar_snap = plt.colorbar(im_snap, label='Temperature [K]', ax=ax_snap)
        cbar_snap.ax.set_ylabel('Temperature [K]', rotation=270, labelpad=20)
        cbar_snap.set_ticks(ticks_k)
        cbar_snap.set_ticklabels(tick_labels)
        ax_snap.set_xlabel('x [mm]')
        ax_snap.set_ylabel('y [mm]')
        
        # Save snapshot
        snapshot_filename = os.path.join(snapshot_folder, f"snapshot_{snapshot_idx+1:03d}_t{t_snapshot:.1f}s.png")
        plt.savefig(snapshot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_snap)
        print(f"Saved snapshot {snapshot_idx+1}/{len(snapshot_times)}: {snapshot_filename}")

# Simulation complete
if not HAS_TQDM:
    print()  # New line after progress indicator
print(f"\nSimulation complete! Generated {len(frames)} frames.")
print(f"Final average air temperature: {inner_temps[-1]-273.15:.2f}°C")
print(f"Final temperature around heat source: {source_temps[-1]-273.15:.2f}°C\n")

# ====================================
# PLOTTING RESULTS
# ====================================
# Calculate animation interval based on playback speed
# Strategy: Use a fast frame rate (10ms) and skip frames to achieve desired speedup
if len(frames) > 0:
    time_per_frame = save_every * dt  # simulation time represented by each frame [s]
    
    # Target animation frame rate - use faster interval for speedup
    # For 60x speedup, we want to show 1 hour in 60 seconds = 1 second per minute of simulation
    target_frame_interval_ms = 10.0  # milliseconds between animation frames (faster for speedup)
    
    # Calculate desired total animation time
    desired_animation_time = t_max / playback_speed_ratio  # seconds
    
    # Calculate how many frames we can show in that time
    max_frames_in_time = int(desired_animation_time * 1000 / target_frame_interval_ms)
    
    # Calculate frame skip to achieve this
    frame_skip = max(1, int(np.ceil(len(frames) / max_frames_in_time))) if max_frames_in_time > 0 else 1
    
    # Actual animation interval
    animation_interval = target_frame_interval_ms
    
    print(f"Each saved frame represents {time_per_frame:.3f} s of simulation time")
    print(f"Desired animation time: {desired_animation_time:.2f} s (to show {t_max:.1f} s at {playback_speed_ratio}x speed)")
    print(f"Frame skip: {frame_skip} (showing every {frame_skip}th frame)")
    print(f"Animation interval: {animation_interval:.1f} ms per frame")
    num_animation_frames = len(frames) // frame_skip + (1 if len(frames) % frame_skip > 0 else 0)
    total_animation_time = num_animation_frames * animation_interval / 1000.0
    actual_speedup = t_max / total_animation_time if total_animation_time > 0 else 1
    print(f"Total animation: {num_animation_frames} frames playing over {total_animation_time:.2f} s")
    print(f"This shows {t_max:.1f} s of simulation at {actual_speedup:.1f}x speed")
else:
    animation_interval = 150  # default
    frame_skip = 1
    num_animation_frames = 0
    print(f"Animation interval: {animation_interval:.1f} ms (no frames to animate)")

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(frames[0], extent=[0,L_total*1000,0,L_total*1000], origin='lower', 
               cmap='inferno', vmin=T_plot_min, vmax=T_plot_max)
plt.title("Temperature field evolution")
cbar = plt.colorbar(im, label='Temperature [K]', ax=ax)
# Add temperature in Celsius on secondary axis
cbar.ax.set_ylabel('Temperature [K]', rotation=270, labelpad=20)
# Add tick marks for key temperatures
cbar.set_ticks(ticks_k)
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
    frame_idx = min(i * frame_skip, len(frames) - 1)  # Ensure we don't exceed available frames
    im.set_data(frames[frame_idx])
    ax.set_title(f"Temperature field at t = {frame_idx*dt*save_every:.1f} s ({frame_idx*dt*save_every/60:.1f} min)")
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=num_animation_frames, interval=int(animation_interval), repeat=True, blit=False)
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
print(f"\nTemperature Evolution:")
print(f"  Final average air temperature: {inner_temps[-1]:.2f} K ({inner_temps[-1]-273.15:.2f}°C)")
print(f"  Final temperature around heat source: {source_temps[-1]:.2f} K ({source_temps[-1]-273.15:.2f}°C)")
print(f"  Maximum temperature around heat source: {max(source_temps):.2f} K ({max(source_temps)-273.15:.2f}°C)")
print(f"\nPhysics Check:")
print(f"  Boundary temp: {T_ext:.1f} K ({T_ext-273.15:.1f}°C)")
print(f"  Final avg air temp: {inner_temps[-1]:.2f} K ({inner_temps[-1]-273.15:.2f}°C)")
temp_rise_due_to_boundary = inner_temps[-1] - T_init
temp_rise_to_boundary = T_ext - T_init
print(f"  Temperature rise: {temp_rise_due_to_boundary:.1f} K ({temp_rise_due_to_boundary:.1f}°C) out of {temp_rise_to_boundary:.1f} K possible")
if temp_rise_to_boundary > 0:
    print(f"  Progress toward boundary temp: {100*temp_rise_due_to_boundary/temp_rise_to_boundary:.1f}%")
    print(f"  Expected: Average temp should approach boundary temp ({T_ext-273.15:.1f}°C)")
    print(f"  The {Q_total}W internal source creates only a local hot spot; boundary heating dominates overall temp.")