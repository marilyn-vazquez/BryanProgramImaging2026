import os
import numpy as np
import matplotlib.pyplot as plt

def generate_curvy_squiggle(start_point, general_direction, total_length, num_points=40, frequency=2.5, amplitude=0.3):
    """Generates a smooth 3D squiggle using a single sine wave in a random plane."""
    start_point = np.array(start_point, dtype=float)
    base_dir = np.array(general_direction, dtype=float)
    base_dir /= np.linalg.norm(base_dir)
    
    if abs(base_dir[2]) < 0.9: 
        perpend_dir = np.cross(base_dir, [0, 0, 1])
    else:
        perpend_dir = np.cross(base_dir, [0, 1, 0])
    perpend_dir /= np.linalg.norm(perpend_dir)
    
    perpend_dir2 = np.cross(base_dir, perpend_dir)
    perpend_dir2 /= np.linalg.norm(perpend_dir2)

    random_plane_angle = np.random.uniform(0, 2 * np.pi)
    wiggle_dir = np.cos(random_plane_angle) * perpend_dir + np.sin(random_plane_angle) * perpend_dir2

    t = np.linspace(0, total_length, num_points)
    cx, cy, cz = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)
    phase_shift = np.random.uniform(0, 2 * np.pi)

    for i in range(num_points):
        straight_pos = start_point + base_dir * t[i]
        wave_phase = (t[i] / total_length) * (frequency * 2 * np.pi)
        wiggle = wiggle_dir * (amplitude * np.sin(wave_phase + phase_shift))
        
        final_pos = straight_pos + wiggle
        cx[i], cy[i], cz[i] = final_pos[0], final_pos[1], final_pos[2]
        
    return cx, cy, cz


# --- Setup Output Folder ---
output_folder = r"C:/Users/ezra.decleene/Documents/BryanProgram2026/Control_Hemisphere"
os.makedirs(output_folder, exist_ok=True)

# --- Editable Parameters ---
num_images = 100
num_lines = 16
radius_x = 4.0
radius_y = 4.0
radius_z = 8.0
min_distance = 3.2  # Strict minimum distance threshold to prevent base overlaps

# Pre-calculate background oval mesh once
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
oval_x = radius_x * np.outer(np.cos(u), np.sin(v))
oval_y = radius_y * np.outer(np.sin(u), np.sin(v))
oval_z = radius_z * np.outer(np.ones(np.size(u)), np.cos(v))

print(f"Generating {num_images} squiggle images with NO overlaps to '{output_folder}'...")

for img_num in range(1, num_images + 1):
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.set_facecolor('black')
    ax.plot_surface(
        oval_x, oval_y, oval_z, 
        color='#333333', 
        alpha=1, 
        edgecolor="none",
        linewidth=0, 
        shade=False,
        zorder=1
    )

    placed_starts = []
    lines_created = 0
    max_attempts = 2000
    attempts = 0

    while lines_created < num_lines and attempts < max_attempts:
        attempts += 1
        
        ui_rand = np.random.uniform(0, 2 * np.pi)
        vi_rand = np.random.uniform(0.1 * np.pi, 0.9 * np.pi)
        
        cand_x = radius_x * np.cos(ui_rand) * np.sin(vi_rand)
        cand_y = radius_y * np.sin(ui_rand) * np.sin(vi_rand)
        cand_z = radius_z * np.cos(vi_rand)
        cand_pt = np.array([cand_x, cand_y, cand_z])
        
        # Hard distance check: Reject completely if closer than min_distance
        too_close = False
        for pt in placed_starts:
            if np.linalg.norm(cand_pt - pt) < min_distance:
                too_close = True
                break
                
        if too_close:
            continue
            
        placed_starts.append(cand_pt)
        lines_created += 1
        
        # Compute pure outward surface normal vectors with slight dampening to avoid trajectory crossing
        dir_x = cand_x / (radius_x**2)
        dir_y = cand_y / (radius_y**2)
        dir_z = cand_z / (radius_z**2)
        
        direction = [
            (dir_x * 0.3) + np.random.uniform(-0.01, 0.01),
            dir_y + np.random.uniform(-0.05, 0.05),
            dir_z + np.random.uniform(-0.05, 0.05)
        ]
        
        rand_length = np.random.uniform(2.1, 2.7)
        rand_freq = np.random.uniform(1.7, 2.3)
        rand_amp = np.random.uniform(0.20, 0.30)  # Tighter wave amplitude to eliminate mid-air crossing
        
        x, y, z = generate_curvy_squiggle(
            start_point=cand_pt,
            general_direction=direction,
            total_length=rand_length,   
            frequency=rand_freq,        
            amplitude=rand_amp          
        )
        
        # Intense white for one half, soft grayish-white for the other half
        line_color = '#FFFFFF' if cand_z > 0 else '#888888'
        
        ax.plot(x, y, z, color=line_color, linewidth=1.2, zorder=10)

    ax.set_box_aspect([1, 1, 2]) 
    ax.view_init(elev=0, azim=0, roll=90)
    ax.grid(False)
    ax.axis('off')

    filepath = os.path.join(output_folder, f"fake_control{img_num:03d}.png")
    plt.savefig(filepath, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

print("Squiggle images generation complete!")
