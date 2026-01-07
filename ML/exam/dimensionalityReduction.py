## %%
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

## %% [markdown]
# # Loading the training and test data

## %% [markdown]
# ## Train data

## %%
train5 = np.loadtxt("mnist_all/train5.txt") / 255  # /255 for normalization
train6 = np.loadtxt("mnist_all/train6.txt") / 255
train8 = np.loadtxt("mnist_all/train8.txt") / 255


## %%
# Define targets
train5_target = 5 * np.ones(len(train5))
train6_target = 6 * np.ones(len(train6))
train8_target = 8 * np.ones(len(train8))

## %%
# Combine data
train_data = np.concatenate([train5, train6, train8])
train_targets = np.concatenate([train5_target, train6_target, train8_target])

## %% [markdown]
# ## Test data

## %%
test5 = np.loadtxt("mnist_all/test5.txt") / 255
test6 = np.loadtxt("mnist_all/test6.txt") / 255
test8 = np.loadtxt("mnist_all/test8.txt") / 255

# Define targets
test5_target = 5 * np.ones(len(test5))
test6_target = 6 * np.ones(len(test6))
test8_target = 8 * np.ones(len(test8))

# Combine
test_data = np.concatenate([test5, test6, test8])
test_targets = np.concatenate([test5_target, test6_target, test8_target])

# Class names
classes = np.unique(test_targets) # [5, 6, 8]

## %% [markdown]
# # Part 1: Reduce dimension to 2
# Here, we wish to reduce the data dimensionality from 784 to 2 using Linear Disicriminant Analysis (LDA).
# For this you can use scikit-learn. The LDA class in scikit-learn fits a covariance matrix and compute eigenvectors for you. LDA assume that you know about the classes, so you have to use the concatenated training set and targets/classes.

## %%
# Fit and Transform a scikit learn LDA instance to training data
transform = LDA(n_components=2).fit(train_data, train_targets)

## %%
# Transform train data from each class using fitted LDA instance
reduced_data = transform.transform(train_data)

print(f"Original: {train_data.shape}; Reduced: {reduced_data.shape}")

## %% [markdown]
# # Part 2: Perform 3-class classification based on the generated 2-dimensional data.
# We need to find a model to classify the test data as either 5, 6, or 8.
# Here, we could use a Gaussian model for each class, and estimate the mean and covariance from the dimensionality reduced data.

## %% [markdown]
# ## Estimate Gaussians using 2-dimensional data obtained from LDA
gmm = GaussianMixture(n_components=len(classes), covariance_type='full',).fit(reduced_data)

## %%
# Estimate parameters for a bivariante Gaussian distribution.
means = gmm.means_              # shape: (3, 2) → mean vector for each class
covariances = gmm.covariances_  # shape: (3, 2, 2) → covariance matrix for each class
weights = gmm.weights_          # prior probability of each class

print(f"Means: {means}\nCovariances: {covariances}\nClass Weights: {weights}")

## %% [markdown]
# ## Classifying test data
# To classify the test data, we first transform it to 2-dimensions as well.

## %%
# Transform test data using fitted LDA instance
reduced_test = transform.transform(test_data)

## %% [markdown]
# Now we compute priors, likelihoods and posteriors

## %%
# Compute priors
priors = gmm.weights_

# Compute Likelihoods
# For each Gaussian, compute pdf value for all test points
likelihoods = np.array([
    norm.pdf(reduced_test, mean=means[k], cov=covariances[k])
    for k in range(len(classes))
])  # shape: (3, n_test)

# Compute posteriors
# Bayes rule: posterior ∝ prior * likelihood
posteriors = priors[:, np.newaxis] * likelihoods

## %% [markdown]
# We can now compute the classification accuracy for the LDA-dimensionality reduced data

## %%
# Compute predictions
pred_labels = classes[np.argmax(posteriors, axis=0)]

print(f"predictions: {pred_labels}")

# Compute accuracy
acc = accuracy_score(test_targets, pred_labels)

print(f"Classification accuracy: {acc*100:.2f}%")

## %% [markdown]
# # Comparison with PCA

## %%
from sklearn.decomposition import PCA

# Fit PCA to training data (unsupervised, doesn't use targets)
pca = PCA(n_components=2)
pca_reduced_data = pca.fit_transform(train_data)
pca_reduced_test = pca.transform(test_data)

print(f"PCA - Original: {train_data.shape}; Reduced: {pca_reduced_data.shape}")

## %% [markdown]
# ## Plot LDA vs PCA comparison

## %%
# Create side-by-side comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color mapping for classes
colors = {5: 'red', 6: 'blue', 8: 'green'}
class_names = {5: 'Digit 5', 6: 'Digit 6', 8: 'Digit 8'}

# Plot LDA
ax = axes[0]
for cls in classes:
    mask = train_targets == cls
    ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
               c=colors[cls], label=class_names[cls], alpha=0.6, s=30)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_title('LDA Dimensionality Reduction')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot PCA
ax = axes[1]
for cls in classes:
    mask = train_targets == cls
    ax.scatter(pca_reduced_data[mask, 0], pca_reduced_data[mask, 1], 
               c=colors[cls], label=class_names[cls], alpha=0.6, s=30)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_title('PCA Dimensionality Reduction')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lda_vs_pca_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved as 'lda_vs_pca_comparison.png'")

## %% [markdown]
# We can see that LDA is better at separating the three classes, while classes 5 and 8 are more overlapped when using PCA.
