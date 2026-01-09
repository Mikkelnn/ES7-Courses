import numpy as np
from sklearn.discriminant_analysis import (
LinearDiscriminantAnalysis as LDA
)
from scipy.stats import norm
x = np.array(
[
[2.5, 2.4],
[0.5, 0.7],
[2.2, 2.9],
[1.9, 2.2],
[3.1, 3.0],
[7.0, 4.5],
[6.0, 5.0],
[7.5, 5.2],
[5.8, 4.8],
[6.2, 5.4],
]
)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
xq = np.array([[1.5, 1.5]])

transform_LDA = LDA(n_components=1).fit(x, y)
print(transform_LDA)
LDA_reduced = transform_LDA.transform(x)
print(LDA_reduced)

def estimate_class_gaussians(reduced_data, targets):
    """
    Estimate Gaussian parameters (mean and covariance) for each class.
    
    Parameters:
    -----------
    reduced_data : ndarray
        Dimensionality-reduced data
    targets : ndarray
        Class labels
    
    Returns:
    --------
    means : dict
        Dictionary mapping class -> mean vector
    covariances : dict
        Dictionary mapping class -> covariance matrix
    """
    means = {}
    covariances = {}
    classes = np.unique(targets)
    
    for cls in classes:
        mask = targets == cls
        class_data = reduced_data[mask]
        means[cls] = np.mean(class_data, axis=0)
        covariances[cls] = np.cov(class_data.T)
    
    print(f"means: {means} covariances: {covariances}")

    return means, covariances

means, covs = estimate_class_gaussians(reduced_data=LDA_reduced, targets=y)

xq_trans = transform_LDA.transform(xq)
print(xq_trans)

pdf0 = norm.pdf(xq_trans, means[0], covs[0])
pdf1 = norm.pdf(xq_trans, means[1], covs[1])
print(f"probality of class 1: {pdf0} Probalility of class 2: {pdf1}")

