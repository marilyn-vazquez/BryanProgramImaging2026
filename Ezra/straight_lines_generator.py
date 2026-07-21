import numpy as np
import matplotlib.pyplot as plt

def generate_straight_line(start_point, general_direction, total_length):
    """Generates a straight 3D line pointing outward from a start point."""
    start_point = np.array(start_point, dtype=float)
    base_dir = np.array(general_direction, dtype=float)
    base_dir /= np.linalg.norm(base_dir) # Normalize direction vector
    
    # Calculate the exact straight endpoint
    end_point = start_point + base_dir * total_length
    
    # Create arrays containing just the start and end coordinates
    cx = np.array([start_point[0], end_point[0]])
    cy = np.array([start_point[1], end_point[1]])
    cz = np.array([start_point[2], end_point[2]])
        
    return cx, cy, cz


# --- Plotting Setup ---
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# --- EDITABLE PARAMETERS ---
num_lines = 16      # <--- Change this number to instantly add or remove lines
radius_x = 4.0      # Oval width
radius_y = 4.0      # Oval depth
radius_z = 8.0      # Oval height (elongated axis)

# Optional seed for random consistency across runs
#np.random.seed(42)

# --- Generate Random Placements ---
for _ in range(num_lines):
    # Pick completely random angles on the 3D ellipsoid surface
    ui_rand = np.random.uniform(0, 2 * np.pi)
    vi_rand = np.random.uniform(0.1 * np.pi, 0.9 * np.pi) # Keeps clear of the tight polar tips
    
    # Calculate the random starting point on the surface of the invisible oval
    start_x = radius_x * np.cos(ui_rand) * np.sin(vi_rand)
    start_y = radius_y * np.sin(ui_rand) * np.sin(vi_rand)
    start_z = radius_z * np.cos(vi_rand)
    
    # Calculate the outward mathematical direction
    dir_x = start_x / (radius_x**2)
    dir_y = start_y / (radius_y**2)
    dir_z = start_z / (radius_z**2)
    
    # Add an organic wobble to the pointing direction vector
    direction = [
        dir_x + np.random.uniform(-0.15, 0.15),
        dir_y + np.random.uniform(-0.15, 0.15),
        dir_z + np.random.uniform(-0.08, 0.08)
    ]
    
    # Randomize the length (Frequency and Amplitude are omitted as lines are straight)
    rand_length = np.random.uniform(2.1, 2.9)
    
    # Generate the straight line pointing outward
    x, y, z = generate_straight_line(
        start_point=[start_x, start_y, start_z],
        general_direction=direction,
        total_length=rand_length
    )
    
    # Plot the straight line
    ax.plot(x, y, z, color='black', linewidth=1.2)

# --- Final Scene Adjustments ---
ax.set_box_aspect([1, 1, 2]) 
ax.view_init(elev=0, azim=0, roll=90)
ax.grid(False)
ax.axis('off')

plt.show()

plt.savefig(
    'C:/Users/ezrad/Downloads/straight_lines.png', 
    dpi=300, 
    transparent=True, 
    bbox_inches='tight', 
    pad_inches=0
)
plt.show()
