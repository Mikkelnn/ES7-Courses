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

exit()

## %% [markdown]
# What does the results show?

## %% [markdown]
# # (Optional Task) Comparison with PCA

## %% [markdown]
# This (optional!) task involves reducing the dimensionality of the data instead using PCA in order to compare it with LDA.

## %%
from sklearn.decomposition import PCA

## %% [markdown]
# ## Part 1

## %% [markdown]
# Optionally also fit The PCA class in scikit-learn fits a covariance matrix and compute eigenvectors for you.
# PCA doesn't assume any knowledge about the classes, so you have to use the concatenated training set.

## %%
# Fit a scikit learn PCA instance to training data

## %% [markdown]
# Now that the PCA model is fit to the training data, we can find a low dimesional representation of each class.

## %% [markdown]
# Let's try to plot the dimensionality reduced data and compare PCA to LDA. What do we see?

## %%
# Scatter plot of the dimensional-reduced data

## %%
# Transform train data from each class using fitted PCA instance

## %% [markdown]
# In the above plot we should see that LDA is seemingly better at seperating the tree classes,while the classes 5 and 8 are highly overlapped when using PCA.

## %% [markdown]
# ## Estimate Gaussians using 2-dimensional data obtained from PCA

## %%
# Estimate parameters for a bivariante Gaussian distribution.

## %% [markdown]
# ## Classifying test data
# To classify the test data, we first transform it to 2-dimensions as well.

## %%
# Transform test data using fitted PCA/LDA instance

## %% [markdown]
# Now we compute priors, likelihoods and posteriors

## %%
# Compute priors
# Compute Likelihoods
# Compute posteriors

## %%
# Compute predictions

# Compute accuracy

## %% [markdown]
# We can now compare the classification accuracy from PCA and LDA. What does the results show?
