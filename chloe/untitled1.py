# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:00:56 2026

@author: chloe
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
import cripser
from gtda.homology import CubicalPersistence
from gtda.plotting import plot_diagram


#X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

#X = X.reshape(-1, 28, 28)
#y = y.astype(int)

#img = X[y == 8][0]

#img = X[y == 8][0] / 255.0
folder= r"C:\Users\chloe\Labs\t10k-images.idx3-ubyte\Control_Stub4_0000.tif"
o_img = Image.open(folder)
arr=np.array(o_img)
cropped=arr[:-300,:]
img = (cropped - cropped.min()) / (cropped.max() - cropped.min())
plt.imshow(img, cmap='gray')
plt.show()

plt.imshow(img, cmap = 'gray')
plt.title("Original Test Image")
plt.axis('off')
plt.show()

ax = sns.heatmap(img, cmap="gray")
ax.set_title("MNIST Digit 8 Heatmap")
plt.show()

cubical = CubicalPersistence(homology_dimensions=(0, 1))
diagrams = cubical.fit_transform(img)


plot_diagram(diagrams[0])
plt.show()

ph = cripser.compute_ph(img.astype(float), maxdim=1)



img_inv = img.max() - img
ph_upper = cripser.compute_ph(img_inv.astype(float), maxdim=1)



finite = np.isfinite(ph[:,2])

df = pd.DataFrame({
    "dimension": ph[finite,0].astype(int),
    "birth": ph[finite,1],
    "death": ph[finite,2]
})


ax = sns.scatterplot(
    data=df,
    x="birth",
    y="death",
    hue="dimension"
)

mn = min(df["birth"].min(), df["death"].min())
mx = max(df["birth"].max(), df["death"].max())

ax.plot([mn, mx], [mn, mx], '--', color='black')

plt.title("Lower Star Persistence")
plt.axis("equal")
plt.show()


finite = np.isfinite(ph_upper[:,2])

df2 = pd.DataFrame({
    "dimension": ph_upper[finite,0].astype(int),
    "birth": ph_upper[finite,1],
    "death": ph_upper[finite,2]
})

ax = sns.scatterplot(
    data=df2,
    x="birth",
    y="death",
    hue="dimension"
)

mn = min(df2["birth"].min(), df2["death"].min())
mx = max(df2["birth"].max(), df2["death"].max())

ax.plot([mn, mx], [mn, mx], '--', color='black')

plt.title("Upper Star Persistence")
plt.axis("equal")
plt.show()

from gtda.homology import WeightedRipsPersistence
from gtda.images import ImageToPointCloud
from gtda.plotting import plot_diagram
from sklearn.neighbors import NearestNeighbors
from skimage.transform import resize
from gtda.images import Binarizer

img_small = resize(
    img,
    (200, 200),
    anti_aliasing=False,
    preserve_range=True,
    order=0
)

img_small = img_small.astype(float)
img_small = (img_small - img_small.min()) / (img_small.max() - img_small.min())


X_img = img_small[np.newaxis, :, :]

binarizer = Binarizer(threshold=0.3)

X_binary = binarizer.fit_transform(X_img)


X_binary = X_binary.astype(int)

#X_binary_small = resize(
 #   X_binary[0],
  #  (128,128),
   # order=0,
    #preserve_range=True,
    #anti_aliasing=False
#)

#X_binary_small = (X_binary_small > 0.5).astype(int)

#print("Points after resize:", np.sum(X_binary_small))

from scipy.spatial import cKDTree

th_test_img = cripser.binarize(cropped, threshold = 100)
plt.imshow(th_test_img, cmap = 'gray')
plt.title("Binarized Test Image")
plt.show()

threshold = 100
rows, cols = np.where(cropped > threshold) # Find the pixels with values larger than the threshold
rows, cols = rows.ravel(), cols.ravel() # Flatten the arrays to 1d
test_img_bin = np.column_stack([rows.astype(np.float64), cols.astype(np.float64)])

x = test_img_bin[:,0]
y = test_img_bin[:,1]
plt.scatter(y, x, s = 0.01, color = "black")
plt.title("Point Cloud of Binarized Test Image")
plt.show()

H, W = cropped.shape # Height and width of original image
test_tree = cKDTree(test_img_bin) # Make an index of nearest neighbors from points in the point cloud
test_rr, test_cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij") # Make an integer grid the same size as the original image
test_queries = np.column_stack([test_rr.ravel().astype(np.float64),
                           test_cc.ravel().astype(np.float64)]) # Shapes the grid into a 2d array, we'll use this to make a grid of values of the DTM filration function


k = 50 # Number of nearest neighbors
test_dists, _ = test_tree.query(test_queries, k = k) # Find the distances to each of the k nearest neighbors for each point in the grid

    
test_dtm_vals = np.sqrt(np.mean(test_dists ** 2, axis = 1)) # Compute the square root of the mean square distance for each point in the grid. This is the DTM filtration function
test_dtm_grid = test_dtm_vals.reshape(H, W) # Reshape into a grid

lo, hi = test_dtm_grid.min(), test_dtm_grid.max() # Normalize the values of the DTM filtration function to [0,1]
if hi > lo:
    test_dtm_grid = (test_dtm_grid - lo) / (hi - lo)

plt.imshow(test_dtm_grid, cmap = 'viridis')
plt.title("Test Image DTM Filtration")
plt.colorbar()
plt.show()



X_pc = ImageToPointCloud().fit_transform(
    X_binary[np.newaxis,:,:]
)

plt.figure(figsize=(6,6))
plt.imshow(X_binary, cmap="gray")

plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")
plt.title("Binarized Resized Image")

plt.xlim(0, X_binary.shape[1])
plt.ylim(X_binary.shape[0], 0)

plt.show()


wrp = WeightedRipsPersistence(
    homology_dimensions=(0, 1),
    n_jobs=-1
)

dtm_diagram = wrp.fit_transform(X_pc)

fig = plot_diagram(dtm_diagram[0])
fig.show(renderer="browser")

X_img = img[np.newaxis, :, :]
binarizer = Binarizer(threshold=0.3)

X_binary = binarizer.fit_transform(X_img)

# convert boolean to float
X_binary = X_binary.astype(float)




img_small = resize(
    X_binary[0],
    (128,128),
    anti_aliasing=True,
    preserve_range=True
)




#X_gray = (X_gray - X_gray.min()) / (X_gray.max() - X_gray.min())

plt.figure(figsize=(6,6))
plt.imshow(img_small, cmap="gray")
plt.title("Cubical Filtration Image")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")
plt.show()

#img_small = (img_small > 0.5).astype(float)

#X_gray = img_small[np.newaxis,:,:]

X_pc = ImageToPointCloud().fit_transform(X_binary[np.newaxis,:,:])

points = X_pc[0]

k = 10
nbrs = NearestNeighbors(n_neighbors=k)
nbrs.fit(points)

distances, _ = nbrs.kneighbors(points)

dtm_values = np.sqrt(np.mean(distances**2, axis=1))

dtm_grid = dtm_values.reshape(img.shape)


wrp = WeightedRipsPersistence(
    homology_dimensions=(0, 1),
    n_jobs=-1
)

diagram = wrp.fit_transform(X_pc)

fig = plot_diagram(diagram[0])
fig.show(renderer="browser")






