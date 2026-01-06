## %%
# ################################### 
# Group ID : 343
# Members : Johan Theil, Marcus Hodal, Mikkel Nielsen, Karl Brandt
# Date : 2025/09/24 
# Lecture: 5 Clustering
# Dependencies: That data set that you gave us
# Python version: 3
# Functionality: Short Description. This script shows overfitting, 
# ridge regression, and sample dependency 
# ################################### 

from scipy.io import loadmat
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib as plt
from sklearn.mixture import GaussianMixture as GMM

## %% [markdown]
# # Exercise 5: Clustering
# This assignment is based on the previously generated 2-dimensional data of the three classes (5, 6 and 8) from the MNIST database of handwritten digits.
#
# First, mix the 2-dimensional data (training data only) by removing the labels and then use one Gaussian mixture model to model them.
#
# Secondly, compare the Gaussian mixture model with the Gaussian models trained in the previous assignment, in terms of mean and variance values as well as through visualisation.

## %% [markdown]
# ## Loading the data and mixing
# First we load the exercise data set, combine the individual training sets into one and shuffle the data to ensure a random shuffle (here with a seed to ensure reproducability).

## %%
data_path = "2D568class.mat"
data = loadmat(data_path)
train5 = data["trn5_2dim"] / 255
train6 = data["trn6_2dim"] / 255
train8 = data["trn8_2dim"] / 255
print("Shape of data:", np.array(data).shape)

# Keep track of class indices before shuffling
n5, n6, n8 = len(train5), len(train6), len(train8)
class_indices_before_shuffle = np.concatenate([
    np.zeros(n5, dtype=int),  # 0 for class 5
    np.ones(n6, dtype=int),   # 1 for class 6
    2 * np.ones(n8, dtype=int)  # 2 for class 8
])

trainset = np.concatenate([train5, train6, train8])
np.random.seed(0)
perm = np.random.permutation(len(trainset))
trainset = trainset[perm]
class_indices = class_indices_before_shuffle[perm]

## %% [markdown]
# ## Creating a Gaussian Mixture model
# First create a Gaussian Mixture Model of the data using sklearn

gmm = GMM(n_components=3)
gmm.fit(trainset)
labels = gmm.predict(trainset)

print(f"GMM fitted with {gmm.n_components} components")
print(f"Weights: {gmm.weights_}")


## %% [markdown]
# ## Creating Gaussian models
# Following the same approach from the previous exercises we can also estimate Gaussian models for each class

## %%
# Estimate means and covariances for each class
mean5 = np.mean(train5, axis=0)
mean6 = np.mean(train6, axis=0)
mean8 = np.mean(train8, axis=0)

cov5 = np.cov(train5.T)
cov6 = np.cov(train6.T)
cov8 = np.cov(train8.T)

print(f"Mean 5: {mean5}")
print(f"Mean 6: {mean6}")
print(f"Mean 8: {mean8}")


## %% [markdown]
# ## Comparing means and covariance matrices.
# Let's look at the means and covariance matrices.
#
# First we extract the means and covariances from the GMM.

## %%
# Extract GMM parameters
gmm_means = gmm.means_
gmm_covariances = gmm.covariances_

mean1_gmm = gmm_means[0]
mean2_gmm = gmm_means[1]
mean3_gmm = gmm_means[2]

cov1_gmm = gmm_covariances[0]
cov2_gmm = gmm_covariances[1]
cov3_gmm = gmm_covariances[2]

print(f"GMM Mean 1: {mean1_gmm}")
print(f"GMM Mean 2: {mean2_gmm}")
print(f"GMM Mean 3: {mean3_gmm}")


## %% [markdown]
# Now we can compare the GMM means and covariances to the Gaussin models estimated for each class individually.

## %% [markdown]
# ### Means

## %%
for name, mean in {"mean5": mean5, "mean6": mean6, "mean8": mean8,
                   "mean1_gmm": mean1_gmm, "mean2_gmm": mean2_gmm, "mean3_gmm": mean3_gmm}.items():
    print(f"{name}: {np.array2string(mean)}")

## %% [markdown]
# ### Covariances

## %%
def plot_covariances(cov5, cov6, cov8, cov1_gmm, cov2_gmm, cov3_gmm):
    """Plot covariance matrices for each class and GMM kernels."""
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].matshow(cov5)
    for (i, j), z in np.ndenumerate(cov5):
        axs[0, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
    axs[0, 0].set_title("Cov. Class: 5")

    axs[1, 0].matshow(cov1_gmm)
    for (i, j), z in np.ndenumerate(cov1_gmm):
        axs[1, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
    axs[1, 0].set_title("Cov. GMM kernel 1")

    axs[0, 1].matshow(cov6)
    for (i, j), z in np.ndenumerate(cov6):
        axs[0, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
    axs[0, 1].set_title("Cov. Class: 6")

    axs[1, 1].matshow(cov2_gmm)
    for (i, j), z in np.ndenumerate(cov2_gmm):
        axs[1, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
    axs[1, 1].set_title("Cov. GMM kernel 2")

    axs[0, 2].matshow(cov8)
    for (i, j), z in np.ndenumerate(cov8):
        axs[0, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
    axs[0, 2].set_title("Cov. Class: 8")

    axs[1, 2].matshow(cov3_gmm)
    for (i, j), z in np.ndenumerate(cov3_gmm):
        axs[1, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
    axs[1, 2].set_title("Cov. GMM kernel 3")

    plt.tight_layout()
    plt.show()

# Call the function
plot_covariances(cov5, cov6, cov8, cov1_gmm, cov2_gmm, cov3_gmm)

## %% [markdown]
# What do we see when comparing means and covariances?

## %% [markdown]
# ## Visualizing the models in contourplots.
# Now we would like to visualize our models to compare them.

## %% [markdown]
# We first generate some points to be able to sample from the models.

## %%
# Create points to do a contour plot
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
points = np.dstack((X, Y))

## %% [markdown]
# We can also visualize the separate Gaussian models from the GMM by creating Gassians from the classwise means and covariances of the GMM model.

## %%
# Sample from individual Gaussians for each class
pdf5 = norm.pdf(points, mean=mean5, cov=cov5)
pdf6 = norm.pdf(points, mean=mean6, cov=cov6)
pdf8 = norm.pdf(points, mean=mean8, cov=cov8)

# Sample from GMM kernels
pdf1_gmm = norm.pdf(points, mean=mean1_gmm, cov=cov1_gmm)
pdf2_gmm = norm.pdf(points, mean=mean2_gmm, cov=cov2_gmm)
pdf3_gmm = norm.pdf(points, mean=mean3_gmm, cov=cov3_gmm)


## %% [markdown]
# Now we sample from the models using the generated points.

## %%
# Compute GMM density
gmm_pdf = np.exp(gmm.score_samples(points.reshape(-1, 2))).reshape(X.shape)


## %% [markdown]
# The model samples can then be visualized in a contour plot.

## %%
def plot_contours(X, Y, pdf5, pdf6, pdf8, pdf1_gmm, pdf2_gmm, gmm_pdf, trainset, class_indices):
    """Plot contours for individual classes and GMM components."""
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # Individual class densities
    axs[0, 0].contour(X, Y, pdf5, levels=10)
    mask5 = class_indices == 0
    axs[0, 0].scatter(trainset[mask5, 0], trainset[mask5, 1], alpha=0.3, s=10, color='red')
    axs[0, 0].set_title("Class 5 Density")
    axs[0, 0].set_xlim(-3, 3)
    axs[0, 0].set_ylim(-3, 3)

    axs[0, 1].contour(X, Y, pdf6, levels=10)
    mask6 = class_indices == 1
    axs[0, 1].scatter(trainset[mask6, 0], trainset[mask6, 1], alpha=0.3, s=10, color='blue')
    axs[0, 1].set_title("Class 6 Density")
    axs[0, 1].set_xlim(-3, 3)
    axs[0, 1].set_ylim(-3, 3)

    axs[0, 2].contour(X, Y, pdf8, levels=10)
    mask8 = class_indices == 2
    axs[0, 2].scatter(trainset[mask8, 0], trainset[mask8, 1], alpha=0.3, s=10, color='green')
    axs[0, 2].set_title("Class 8 Density")
    axs[0, 2].set_xlim(-3, 3)
    axs[0, 2].set_ylim(-3, 3)

    # GMM kernel densities
    axs[1, 0].contour(X, Y, pdf1_gmm, levels=10)
    axs[1, 0].set_title("GMM Kernel 1 Density")
    axs[1, 0].set_xlim(-3, 3)
    axs[1, 0].set_ylim(-3, 3)

    axs[1, 1].contour(X, Y, pdf2_gmm, levels=10)
    axs[1, 1].set_title("GMM Kernel 2 Density")
    axs[1, 1].set_xlim(-3, 3)
    axs[1, 1].set_ylim(-3, 3)

    axs[1, 2].contour(X, Y, gmm_pdf, levels=10)
    axs[1, 2].scatter(trainset[:, 0], trainset[:, 1], alpha=0.3, s=10)
    axs[1, 2].set_title("Combined GMM Density")
    axs[1, 2].set_xlim(-3, 3)
    axs[1, 2].set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()

# Call the function
# plot_contours(X, Y, pdf5, pdf6, pdf8, pdf1_gmm, pdf2_gmm, gmm_pdf, trainset, class_indices)

## %%
def classify_and_compare(gmm, test_data, true_labels, class_mapping=None):
    """
    Classify test data using GMM and compare with true labels.
    
    Parameters:
    -----------
    gmm : GaussianMixture
        Fitted Gaussian Mixture Model
    test_data : ndarray
        Test data (n_samples, n_features)
    true_labels : ndarray
        True class labels (0, 1, 2 for classes 5, 6, 8)
    class_mapping : dict, optional
        Mapping from component index to class label (e.g., {0: 5, 1: 6, 2: 8})
    
    Returns:
    --------
    confusion_matrix : ndarray
        Confusion matrix (true_label, predicted_label)
    accuracy : float
        Classification accuracy
    """
    from sklearn.metrics import confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    
    # Predict labels using GMM
    predicted_component = gmm.predict(test_data)
    
    # If class_mapping is not provided, map 0->5, 1->6, 2->8
    if class_mapping is None:
        class_mapping = {0: 5, 1: 6, 2: 8}
    
    # Convert component predictions to class labels
    predicted_labels = np.array([class_mapping.get(c, c) for c in predicted_component])
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[5, 6, 8])
    
    # Compute accuracy
    acc = accuracy_score(true_labels, predicted_labels)
    
    # Print results
    print("Confusion Matrix:")
    print("Rows: True labels (5, 6, 8)")
    print("Cols: Predicted labels (5, 6, 8)")
    print(cm)
    print(f"\nClassification Accuracy: {acc*100:.2f}%")
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Set labels
    labels = ['5', '6', '8']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    
    return cm, acc

# Test data needs to be loaded for classification
# Uncomment and run if you have test data available
test5 = data["tst5_2dim"] / 255
test6 = data["tst6_2dim"] / 255
test8 = data["tst8_2dim"] / 255
test_data = np.concatenate([test5, test6, test8])
test_labels = np.concatenate([
    5 * np.ones(len(test5), dtype=int),
    6 * np.ones(len(test6), dtype=int),
    8 * np.ones(len(test8), dtype=int)
])
cm, acc = classify_and_compare(gmm, test_data, test_labels)

## %%
