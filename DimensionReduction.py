# AMATH 582 - Homework 2 - Dimension Reduction
# Sukhjit Kaur

# Import packages
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

## TASK 1 - create X_matrix ##

folder_path = '/Users/sukhjitkaur/Documents/AMATH582/Homework 2/hw2data/train'
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
# print("Found files:", file_list)  # Debugging

# Initialize empty list
data_list = []

# Load each file and append to data_list
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path)  # Shape should be (114, 100) 38*3
    data_list.append(data)

# Convert list to a NumPy array
X_train = np.array(data_list)  # (15, 114, 100)
# print(X_train.shape)  # Debugging

# Reshape to (15 * 100, 114) so  PCA will treat spatial coordinates as features
X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, 114)  # (1500, 114) each row is timestep, and each column is joint coordinate

# print("Reshaped X_train shape:", X_train_reshaped.shape)  # Debugging

# Use Principal Component Analysis
pca = PCA()
X_pca = pca.fit_transform(X_train_reshaped)  # Shape: (1500, 114)

# Explained variance ratio
explained_var = pca.explained_variance_ratio_

# Compute cumulative variance
cumulative_var = np.cumsum(explained_var)

# Commented out plot code
# Plot cumulative variance
# plt.figure(figsize=(8,5))
# plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='*', linestyle='-')
# plt.xlabel('Number of PCA Modes')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Cumulative Energy Plot')
# plt.axhline(y=0.7, color='r', linestyle='--', label='70% variance')
# plt.axhline(y=0.8, color='b', linestyle='--', label='80% variance')
# plt.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
# plt.axhline(y=0.95, color='magenta', linestyle='--', label='95% variance')
# plt.legend()
# plt.grid()
# plt.show()

## TASK 2 - truncate modes and visualize ##

# Find number of modes needed for different thresholds
thresholds = [0.7, 0.8, 0.9, 0.95] # 70, 80, 90, 95% thresholds
modes_needed = [np.argmax(cumulative_var >= t) + 1 for t in thresholds]

# Commented out print lines
# for t, m in zip(thresholds, modes_needed):
#     print(f"Number of PCA modes needed for {t*100:.0f}% variance: {m}")

# PCA with 2 components
pca_2 = PCA(n_components=2) # 2D
X_pca_2 = pca_2.fit_transform(X_train_reshaped)

# PCA with 3 components
pca_3 = PCA(n_components=3) # 3D
X_pca_3 = pca_3.fit_transform(X_train_reshaped)

# Ground truth labels -  0 = walking, 1 = running, 2 = jumping
y_train = np.array([0] * 500 + [1] * 500 + [2] * 500)  # 5 samples per movement type, each with 100 timesteps

# Plot for 2D trajectories

# Define colors for each movement
colors = ['blue', 'green', 'red']
labels = ['Walking', 'Jumping', 'Running']

# Commented out plot code
# plt.figure(figsize=(8,6))
# for i, label in enumerate(labels):
#     plt.scatter(X_pca_2[y_train == i, 0], X_pca_2[y_train == i, 1],
#                 color=colors[i], alpha=0.5, label=label)
#
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("2-D PCA Projection of Movements")
# plt.legend()
# plt.grid()
# plt.show()

# Plot for 3D trajectories

# Commented out plot code
# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
#
# for i, label in enumerate(labels):
#     ax.scatter(X_pca_3[y_train == i, 0], X_pca_3[y_train == i, 1], X_pca_3[y_train == i, 2],
#                color=colors[i], alpha=0.5, label=label)
#
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# ax.set_title("3-D PCA Projection of Movements")
# ax.legend()
# plt.show()

## TASK 3 - centroids in k-modes PCA space, various k ##

# k = number of PCA components to use for classification
k_list = [2, 3, 4, 5, 8, 10, 16, 20]  # k = 3 for 3D space, k = 2 for 2D

# K components PCA
for k in k_list:
    pca_k = PCA(n_components=k)
    X_pca_k = pca_k.fit_transform(X_train_reshaped)

    # Compute centroid for each class (0 = walking, 1 = jumping, 2 = running)
    centroids = []
    for i in range(3):  # 3 movements
        centroids.append(np.mean(X_pca_k[y_train == i], axis=0))

    centroids = np.array(centroids)
    #print(f"Centroids in {k} PCA space:\n", centroids)

    ## TASK 4 - accuracy ##

    #   Assign labels based on nearest centroid
    def assign_labels(X_pca_k, centroids):
        labels = []
        for sample in X_pca_k:
            distances = np.linalg.norm(sample - centroids, axis=1)  # Find distance to each centroid
            labels.append(np.argmin(distances))  # Assign class of the nearest centroid
        return np.array(labels)

    # Get predicted labels for training data
    y_train_pred = assign_labels(X_pca_k, centroids)

    # Compute accuracy of classification
    train_acc = accuracy_score(y_train, y_train_pred)
    #print(f"Training Accuracy is {train_acc * 100:.2f}% for {k}-mode")


    ## TASK 5 - predict test labels ##

    # Load test samples (assuming test data is stored similarly)
    folder_path_test = '/Users/sukhjitkaur/Documents/AMATH582/Homework 2/hw2data/test'
    file_list_test = sorted([f for f in os.listdir(folder_path_test) if f.endswith(".npy")])
    # print("Found files:", file_list_test)  # Debugging
    # Initialize list
    test_list = []

    # Load each file and append to data_list
    for file_name in file_list_test:
        file_path_test = os.path.join(folder_path_test, file_name)
        data_test = np.load(file_path_test)
        test_list.append(data_test)

    # Convert list to a NumPy array
    X_test = np.array(test_list)
    # print(X_test.shape)  # Debugging

    X_test_reshaped = X_test.transpose(0, 2, 1).reshape(-1, 114)

    # Apply PCA transformation from training data
    X_test_pca_k = pca_k.transform(X_test_reshaped)  # Transform test data using PCA train model

    # Ground truth labels for test
    y_test = np.repeat([0, 1, 2], 100)  # 100 test sample per type to match train set

    # Predict test labels with nearest centroid
    y_test_pred = assign_labels(X_test_pca_k, centroids)

    # Find test accuracy
    test_acc = accuracy_score(y_test, y_test_pred)
    #print(f"Test Accuracy is {test_acc * 100:.2f}% for {k}-mode")

## Bonus Task - implement K-NN ##

# PCA components to use
kn = 3  # 3-D

#  PCA on training data
pca_knn = PCA(n_components=kn)
X_pca_knn = pca_knn.fit_transform(X_train_reshaped)

# Train k nearest neighbor classifier
knn = KNeighborsClassifier(n_neighbors=3)  # 3 nearest neighbors
knn.fit(X_pca_knn, y_train)

# Predict on training data
y_pred_knn_train = knn.predict(X_pca_knn)
train_acc_knn = accuracy_score(y_train, y_pred_knn_train)
#print(f"k-NN Training Accuracy: {train_acc_knn * 100:.2f}% for 3-mode with 3 nearest neighbors")

# Transform test data with the PCA transformation
X_test_pca_knn = pca_knn.transform(X_test_reshaped)

# Predict on test data
y_pred_knn_test = knn.predict(X_test_pca_knn)
test_acc_knn = accuracy_score(y_test, y_pred_knn_test)
#print(f"k-NN Test Accuracy: {test_acc_knn * 100:.2f}% for 3-mode with 3 nearest neighbors")

## From Homework Helper, for visualization of data
# fname= "walking_1"
# folder = "hw2datanpy/train/"
#
# vals = np.load(folder+fname+".npy")
# xyz = np.reshape( vals[:,:], (38,3,-1) )
#
#
# print(xyz.shape)
#
# #define the root joint and scaling of the values
# r = 1000
# xroot, yroot, zroot = xyz[0,0,0], xyz[0,0,1], xyz[0,0,2]
#
# #define the connections between the joints (skeleton)
# I = np.array(
#         [1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31,
#          32, 33, 34, 35, 33, 37]) - 1
# J = np.array(
#         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
#          33, 34, 35, 36, 37, 38]) - 1
