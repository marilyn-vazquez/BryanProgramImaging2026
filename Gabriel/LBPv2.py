# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern 
from skimage.measure import block_reduce
from PIL import Image 
from ripser import ripser
from persim import plot_diagrams

img = Image.open(
    r"C:\Users\g_gar\OneDrive - Simpson College\BryanSummer\Control_Stub4_0000 .tif"
).convert("L")


width, height = img.size
pixels_to_remove = 296
cropped_border_img = img.crop((0, 0, width, height - pixels_to_remove))

plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(cropped_border_img, cmap = 'gray')

img_array = np.array(cropped_border_img)

radius = 2
n_points = 8 * radius

lbp = local_binary_pattern(img_array, n_points, radius, method = "uniform")

lbp_crop = lbp[450:3500, 0:5000]

#Figure 1
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(lbp_crop, cmap = 'gray')

binary_img = (lbp == 17).astype(np.int8)

# print(binary_img.shape)

#Figure 2
plt.figure(figsize = (8,8))
plt.axis('off')
plt.title("Pattern 17 Binarized Image")
plt.imshow(binary_img, cmap = 'gray')

downsample_factor = 20

binary_small = block_reduce(
    binary_img,
    block_size=(downsample_factor, downsample_factor),
    func = np.mean
)

threshold = 0.48
binary_small_thresh = (binary_small < threshold).astype(np.uint8)
print(binary_small_thresh.shape)

# Figure 3
plt.figure(figsize=(8,8))
plt.imshow(binary_small_thresh, cmap="gray")
plt.title("Downsampled + Thresholded Binary Image")
plt.axis("off")
plt.show()

# Point cloud 
points = np.column_stack(np.where(binary_small_thresh == 1))


max_points = 4000

np.random.seed(42)

if points.shape[0] > max_points:
    idx = np.random.choice(points.shape[0], max_points, replace = False)
    points_sample = points[idx]
else:
    points_sample = points

# Figure 4
plt.figure(figsize = (8,8))
plt.scatter(points_sample[:,1], points_sample[:,0], s = 7)
plt.axis('equal')
plt.title("Sample Point Cloud")
plt.gca().invert_yaxis()
plt.show()

# Figure 5
result = ripser(points_sample, maxdim = 1, thresh = 30)
diagrams = result["dgms"]

plot_diagrams(diagrams, show = True)

