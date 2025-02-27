# AMATH 582 - Homework 3 - MNIST Classifier
# Sukhjit Kaur

# Import packages and libraries
import struct
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_score, cross_val_predict

## TASK 1 - PCA analysis and plot first 16 modes ##
with open('/Users/sukhjitkaur/Documents/AMATH582/Homework 3/data/train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    #print(data.shape)
    Xtraindata = data.reshape((size, nrows*ncols))

with open('/Users/sukhjitkaur/Documents/AMATH582/Homework 3/data/train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytrainlabels = data.reshape((size,)) # (Optional)

with open('/Users/sukhjitkaur/Documents/AMATH582/Homework 3/data/t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtestdata = data.reshape((size, nrows*ncols))

with open('/Users/sukhjitkaur/Documents/AMATH582/Homework 3/data/t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytestlabels = data.reshape((size,)) # (Optional)

#train_img = np.transpose(Xtraindata).reshape((60000,28,28))
# print(Xtraindata.shape)
# print(ytrainlabels.shape)
# print(Xtestdata.shape)
# print(ytestlabels.shape)

pca = PCA() #n_components = pca modes kept, first keep all
Xtraindata_pca = StandardScaler().fit_transform(pca.fit_transform(Xtraindata))
components = pca.components_

# plotting
# function to plot the MNIST digits (hw3 helper)
def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(10, 10))

    for i in range(N):
        for j in range(N):
            ax[i, j].imshow(XX[(N) * i + j,:].reshape((28, 28)), cmap="Greys")
            ax[i, j].axis("off")
    fig.suptitle(title, fontsize=24)

# function to plot confusion matrices later
def plot_confusion_matrix(cm, labels, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar = False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=18)
    plt.show()

# Plot the first 64 training images and the first 16 modes
plot_digits(Xtraindata, 8, "First 64 Training Images from MNIST Data")
plt.show()
plot_digits(components[:16,:], 4, "First 16 Principal Component Modes")
plt.show()

## TASK 2 - Find k PC to approx 85% of energy ##
# Explained variance ratio
explained_var = pca.explained_variance_ratio_
#print(explained_var)
# Compute cumulative variance
cumulative_var = np.cumsum(explained_var)
#print(cumulative_var)

#Plot cumulative variance and explained var ratio
plt.plot(explained_var, color = 'blue', linewidth = 3, linestyle = '--')
plt.xticks(); plt.yticks()
plt.xlabel('PCA Component', fontsize = 10)
plt.ylabel('Explained Variance Ratio', fontsize = 10)
plt.title('Explained Variance Ratio for N = 784 Principal Components', fontsize = 12)
plt.grid()
plt.show()

plt.figure()
plt.plot(np.arange(1, len(cumulative_var) + 1), cumulative_var, linewidth = 3, linestyle='-')
plt.xticks(); plt.yticks()
plt.xlabel('PCA Component', fontsize = 10)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize = 10)
plt.title('Cumulative Explained Variance Ratio for N = 784 Principal Components', fontsize = 12)
plt.axhline(y=0.7, color='r', linestyle='--', label='70% variance')
plt.axhline(y=0.85, color='b', linestyle='--', label='85% variance')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
plt.axhline(y=0.95, color='magenta', linestyle='--', label='95% variance')
plt.legend()
plt.grid()
plt.show()

# Find number of modes needed for different thresholds
thresholds = [0.7, 0.85, 0.9, 0.95] # 70, 85, 90, 95% thresholds
modes_needed = [np.argmax(cumulative_var >= t) + 1 for t in thresholds]
for t, m in zip(thresholds, modes_needed):
    print(f"Number of PCA modes needed for {t*100:.0f}% variance: {m}")

# truncate and visualize reconstruction for 85% variance
n = modes_needed[1]  # for 85% var
n_img = 5
n_samples, n_features = Xtraindata.shape
estimator = decomposition.PCA(n_components=n, svd_solver='randomized', whiten=True)
Xtrain_recon = estimator.inverse_transform(estimator.fit_transform(Xtraindata))
idx = np.random.choice(n_samples, n_img, replace=False)

plt.figure()
for i in range(len(idx)):
    plt.subplot(1,n_img,i+1), plt.imshow(np.reshape(Xtraindata[idx[i],:], (28,28))), plt.axis('off')
plt.suptitle('Original', size=20)
plt.show()

plt.figure()
for i in range(len(idx)):
    plt.subplot(1,n_img,i+1), plt.imshow(np.reshape(Xtrain_recon[idx[i],:], (28,28))), plt.axis('off')
plt.suptitle(f"Reconstructed with {n} Principal Components".format(n), size=20)
plt.show()

## TASK 3 - Select subset of digits ##

def select_digits(X, Y, dig1, dig2):
    # Get indices of digits a and b
    Y_dig1 = np.where(Y == dig1)[0]  # Getting the actual indices as 1D array
    Y_dig2 = np.where(Y == dig2)[0]

    # Select features corresponding to digits a and b
    X_dig1 = X[Y_dig1, :]
    X_dig2 = X[Y_dig2, :]

    # Concatenate features from a and b
    X_subset = np.concatenate((X_dig1, X_dig2), axis=0)

    # Create labels for a and b
    Y_subset = np.concatenate((
        np.full(len(Y_dig1), -1),  # Label a as -1
        np.full(len(Y_dig2), 1)    # Label b as 1
    ))

    return X_subset, Y_subset

## TASK 4 - Train classifier with 1,8 pair ##
X_subtrain_18, y_subtrain_18 = select_digits(Xtraindata_pca, ytrainlabels, 1,8)
X_subtest_18, y_subtest_18 = select_digits(Xtestdata, ytestlabels, 1,8)

# print(f"Original X_subtrain_18 shape: {X_subtrain_18.shape}")
# print(f"Original y_subtrain_18 shape: {y_subtrain_18.shape}")

# Project the subset onto the PC space
k = n  # for 85% energy
X_subtrain_18_proj = pca.transform(X_subtrain_18)[:, :k]
X_subtest_18_proj = pca.transform(X_subtest_18)[:, :k]

# print("X_subtrain_18_proj shape:", X_subtrain_18_proj.shape)
# print("y_subtrain_18 shape:", y_subtrain_18.shape)

# Train ridge classifier
ridge = RidgeClassifierCV(alphas= (0.01, 0.1, 1.0), cv=5)

# Fit the classifier
ridge.fit(X_subtrain_18_proj, y_subtrain_18)

# Predictions for training and scores
train_predict_18 = ridge.predict(X_subtrain_18_proj)
train_pred_acc_18 = accuracy_score(y_subtrain_18, train_predict_18)

#Predictions for test and scores
test_predict_18 = ridge.predict(X_subtest_18_proj)
test_pred_acc_18 = accuracy_score(y_subtest_18, test_predict_18)
mse_18_test = mean_squared_error(y_subtest_18, test_predict_18)

# Cross-validation
cv_scores_18 = cross_val_score(ridge, X_subtrain_18_proj, y_subtrain_18, cv=5)
mean_cv_score_18 = cv_scores_18.mean()
std_cv_score_18 = cv_scores_18.std()

cm_18 = confusion_matrix(y_subtest_18, test_predict_18)
plot_confusion_matrix(cm_18, ['1', '8'], "Confusion Matrix - Digits 1 and 8")

# Display results for 1,8 pair
print(f"Cross-validation accuracy for 1,8: {mean_cv_score_18:.4f} ± {std_cv_score_18:.4f}")
print(f"Training accuracy for 1,8: {100*train_pred_acc_18:.4f}%")
print(f"Test accuracy for 1,8: {100*test_pred_acc_18:.4f}%")



## TASK 5 - Train classifier with 3,8 and 2,7 pairs ##
X_subtrain_38, y_subtrain_38 = select_digits(Xtraindata_pca, ytrainlabels, 3,8)
X_subtest_38, y_subtest_38 = select_digits(Xtestdata, ytestlabels, 3,8)

# print(f"Original X_subtrain_18 shape: {X_subtrain_18.shape}")
# print(f"Original y_subtrain_18 shape: {y_subtrain_18.shape}")

# Project the subset onto the PC space
k = n
X_subtrain_38_proj = pca.transform(X_subtrain_38)[:, :k]
X_subtest_38_proj = pca.transform(X_subtest_38)[:, :k]

# print("X_subtrain_18_proj shape:", X_subtrain_18_proj.shape)
# print("y_subtrain_18 shape:", y_subtrain_18.shape)

# Train ridge classifier
ridge = RidgeClassifierCV(alphas= (0.01, 0.1, 1.0), cv=5)

# Fit the classifier
ridge.fit(X_subtrain_38_proj, y_subtrain_38)

train_predict_38 = ridge.predict(X_subtrain_38_proj)
train_pred_acc_38 = accuracy_score(y_subtrain_38, train_predict_38)

#Predictions for test and scores
test_predict_38 = ridge.predict(X_subtest_38_proj)
test_pred_acc_38 = accuracy_score(y_subtest_38, test_predict_38)

# Cross-validation
cv_scores_38 = cross_val_score(ridge, X_subtrain_38_proj, y_subtrain_38, cv=5)
mean_cv_score_38 = cv_scores_38.mean()
std_cv_score_38 = cv_scores_38.std()

cm_38 = confusion_matrix(y_subtest_38, test_predict_38)
plot_confusion_matrix(cm_38, ['3', '8'],"Confusion Matrix - Digits 3 and 8 ")

# Display results for 3,8 pair
print(f"Cross-validation accuracy for 3,8: {mean_cv_score_38:.4f} ± {std_cv_score_38:.4f}")
print(f"Training accuracy for 3,8: {100*train_pred_acc_38:.4f}%")
print(f"Test accuracy for 3,8: {100*test_pred_acc_38:.4f}%")

# For 2,7 pair
X_subtrain_27, y_subtrain_27 = select_digits(Xtraindata_pca, ytrainlabels, 2,7)
X_subtest_27, y_subtest_27 = select_digits(Xtestdata, ytestlabels, 2,7)

# print(f"Original X_subtrain_18 shape: {X_subtrain_18.shape}")
# print(f"Original y_subtrain_18 shape: {y_subtrain_18.shape}")

# Project the subset onto the PC space
k = n  # for 85% energy
X_subtrain_27_proj = pca.transform(X_subtrain_27)[:, :k]
X_subtest_27_proj = pca.transform(X_subtest_27)[:, :k]

# print("X_subtrain_18_proj shape:", X_subtrain_18_proj.shape)
# print("y_subtrain_18 shape:", y_subtrain_18.shape)

# Train ridge classifier
ridge = RidgeClassifierCV(alphas= (0.01, 0.1, 1.0), cv=5)

# Fit the classifier
ridge.fit(X_subtrain_27_proj, y_subtrain_27)

# Predictions for training and scores
train_predict_27 = ridge.predict(X_subtrain_27_proj)
train_pred_acc_27 = accuracy_score(y_subtrain_27, train_predict_27)

#Predictions for test and scores
test_predict_27 = ridge.predict(X_subtest_27_proj)
test_pred_acc_27 = accuracy_score(y_subtest_27, test_predict_27)

# Cross-validation
cv_scores_27 = cross_val_score(ridge, X_subtrain_27_proj, y_subtrain_27, cv=5)
mean_cv_score_27 = cv_scores_27.mean()
std_cv_score_27 = cv_scores_27.std()

cm_27 = confusion_matrix(y_subtest_27, test_predict_27)
plot_confusion_matrix(cm_27, ['2', '7'],"Confusion Matrix - Digits 2 and 7")

# Display results for 2,7 pair
print(f"Cross-validation accuracy for 2,7: {mean_cv_score_27:.4f} ± {std_cv_score_27:.4f}")
print(f"Training accuracy for 2,7: {100*train_pred_acc_27:.4f}%")
print(f"Test accuracy for 2,7: {100*test_pred_acc_27:.4f}%")

# TASK 6 - other classifiers ##
# Project training and test data onto k PCA components (85% energy)
k = n
Xtrain_proj = pca.transform(Xtraindata)[:, :k]
Xtest_proj = pca.transform(Xtestdata)[:, :k]

# Ridge Classifier
ridge_multi = RidgeClassifier(alpha=1.0)
ridge_multi.fit(Xtrain_proj, ytrainlabels)
ridge_train_acc = ridge_multi.score(Xtrain_proj, ytrainlabels)
ridge_test_acc = ridge_multi.score(Xtest_proj, ytestlabels)
# Ridge Regression Confusion Matrix
ridge_preds = ridge_multi.predict(Xtest_proj)
ridge_cm = confusion_matrix(ytestlabels, ridge_preds)
print(f"Ridge Classifier - Training accuracy: {100*ridge_train_acc:.4f}%")
print(f"Ridge Classifer - Test accuracy: {100*ridge_test_acc:.4f}%")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain_proj, ytrainlabels)
knn_train_acc = knn.score(Xtrain_proj, ytrainlabels)
knn_test_acc = knn.score(Xtest_proj, ytestlabels)
# KNN confusion matrix
knn_preds = knn.predict(Xtest_proj)
knn_cm = confusion_matrix(ytestlabels, knn_preds)
print(f"KNN - Training accuracy: {100*knn_train_acc:.4f}%")
print(f"KNN - Test accuracy: {100*knn_test_acc:.4f}%")

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(Xtrain_proj, ytrainlabels)
lda_train_acc = lda.score(Xtrain_proj, ytrainlabels)
lda_test_acc = lda.score(Xtest_proj, ytestlabels)
# LDA confusion matrix
lda_preds = lda.predict(Xtest_proj)
lda_cm = confusion_matrix(ytestlabels, lda_preds)
print(f"LDA - Training accuracy: {100*lda_train_acc:.4f}%")
print(f"LDA - Test accuracy: {100*lda_test_acc:.4f}%")

#plot confusion matrices for the multi-class classification
matrices = [ridge_cm, knn_cm, lda_cm]
titles = ["Ridge Classifier", "KNN", "LDA"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (title, matrix) in enumerate(zip(titles, matrices)):
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {title}')

plt.tight_layout()
plt.show()
