# %%
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import svm

# %%
def create_complete_datasets(data_dict):
    '''
    Function for creating complete training and test sets containing
    all classes.
    '''
    #Empty list
    trainset = []
    traintargets =[]
    testset = []
    testtargets =[]
    
    #For each class
    for i in range(10):
        trainset.append(data_dict["train%d"%i])
        traintargets.append(np.full(len(data_dict["train%d"%i]),i))
        testset.append(data_dict["test%d"%i])
        testtargets.append(np.full(len(data_dict["test%d"%i]),i))
    
    #Concatenate into to complete datasets
    trainset = np.concatenate(trainset)
    traintargets = np.concatenate(traintargets)
    testset = np.concatenate(testset)
    testtargets = np.concatenate(testtargets)
    return trainset, traintargets, testset, testtargets

file = "mnist_all.mat"
data = loadmat(file)

#Complete training and test sets
train_set, train_targets, test_set, test_targets = create_complete_datasets(data)

# %%
n_components = 2

#PCA
transform_pca = PCA(n_components=n_components).fit(train_set)
pca_reduced_train = transform_pca.transform(train_set)

#LDA
transform_LDA = LDA(n_components=n_components).fit(train_set, train_targets)
lda_reduced_train = transform_LDA.transform(train_set)

#SVM on PCA
svm_pca_model = svm.SVC(kernel='rbf').fit(pca_reduced_train, train_targets)

#SVM on LDA
svm_lda_model = svm.SVC(kernel='rbf').fit(lda_reduced_train, train_targets)


# %%
# Analyze proportion of Variance. If num_components=2 try to visualize dim. reduced data.

## %%
def plot_pca_lda_comparison(train_set, train_targets, transform_pca, transform_lda):
    """Plot PCA and LDA reduced data side by side for comparison."""
    
    # Transform training data
    pca_reduced = transform_pca.transform(train_set)
    lda_reduced = transform_lda.transform(train_set)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot PCA
    ax = axes[0]
    scatter_pca = ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1], 
                             c=train_targets, cmap='tab10', alpha=0.6, s=20)
    ax.set_xlabel(f'PC 1 ({transform_pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC 2 ({transform_pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Dimensionality Reduction')
    ax.grid(True, alpha=0.3)
    cbar_pca = plt.colorbar(scatter_pca, ax=ax)
    cbar_pca.set_label('Class')
    
    # Plot LDA
    ax = axes[1]
    scatter_lda = ax.scatter(lda_reduced[:, 0], lda_reduced[:, 1], 
                             c=train_targets, cmap='tab10', alpha=0.6, s=20)
    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')
    ax.set_title('LDA Dimensionality Reduction')
    ax.grid(True, alpha=0.3)
    cbar_lda = plt.colorbar(scatter_lda, ax=ax)
    cbar_lda.set_label('Class')
    
    plt.tight_layout()
    plt.show()
    
    print(f"PCA - Explained variance ratio: {transform_pca.explained_variance_ratio_}")
    print(f"PCA - Cumulative variance explained: {np.sum(transform_pca.explained_variance_ratio_)*100:.2f}%")

# Call the plotting function
# plot_pca_lda_comparison(train_set, train_targets, transform_pca, transform_LDA)


# %%
# Estimate Gaussians from PCA/LDA

## %%
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
    
    return means, covariances



# Estimate Gaussians for each method
pca_means, pca_covs = estimate_class_gaussians(pca_reduced_train, train_targets)
lda_means, lda_covs = estimate_class_gaussians(lda_reduced_train, train_targets)

print("Gaussians estimated for PCA and LDA")
print(f"Number of classes: {len(pca_means)}")

# %%
def classify_with_gaussians(reduced_data, means, covariances):
    """
    Classify data using Gaussian models (Bayes classifier).
    
    Parameters:
    -----------
    reduced_data : ndarray
        Dimensionality-reduced data (n_samples, n_components)
    means : dict
        Dictionary mapping class -> mean vector
    covariances : dict
        Dictionary mapping class -> covariance matrix
    
    Returns:
    --------
    predictions : ndarray
        Predicted class labels
    """
    n_samples = reduced_data.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        max_prob = -np.inf
        best_class = None
        
        for cls in means.keys():
            try:
                # Compute Gaussian PDF value
                pdf_val = norm.pdf(reduced_data[i], mean=means[cls], cov=covariances[cls])
                if pdf_val > max_prob:
                    max_prob = pdf_val
                    best_class = cls
            except:
                # Handle singular covariance matrix
                continue
        
        predictions[i] = best_class if best_class is not None else 0
    
    return predictions

#Compute predictions
pca_predictions = classify_with_gaussians(transform_pca.transform(test_set), pca_means, pca_covs)
lda_predictions = classify_with_gaussians(transform_LDA.transform(test_set), lda_means, lda_covs)
svm_pca_predictions = svm_pca_model.predict(transform_pca.transform(test_set))
svm_lda_predictions = svm_lda_model.predict(transform_LDA.transform(test_set))

print("Predictions computed")

#Compute accuracy
from sklearn.metrics import accuracy_score

pca_accuracy = accuracy_score(test_targets, pca_predictions)
lda_accuracy = accuracy_score(test_targets, lda_predictions)
svm_pca_accuracy = accuracy_score(test_targets, svm_pca_predictions)
svm_lda_accuracy = accuracy_score(test_targets, svm_lda_predictions)

print(f"\nPCA Accuracy: {pca_accuracy*100:.2f}%")
print(f"LDA Accuracy: {lda_accuracy*100:.2f}%")
print(f"SVM (PCA) Accuracy: {svm_pca_accuracy*100:.2f}%")
print(f"SVM (LDA) Accuracy: {svm_lda_accuracy*100:.2f}%")

# %%
#Compute the confusion matrices for PCA, LDA, SVM(PCA), and SVM(LDA)
pca_cm = confusion_matrix(test_targets, pca_predictions)
lda_cm = confusion_matrix(test_targets, lda_predictions)
svm_pca_cm = confusion_matrix(test_targets, svm_pca_predictions)
svm_lda_cm = confusion_matrix(test_targets, svm_lda_predictions)

#Plot Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# PCA Confusion Matrix
disp_pca = ConfusionMatrixDisplay(confusion_matrix=pca_cm, display_labels=np.arange(10))
disp_pca.plot(ax=axes[0, 0], cmap='Blues', values_format='d')
axes[0, 0].set_title(f'PCA + Gaussian\nAccuracy: {pca_accuracy*100:.2f}%')

# LDA Confusion Matrix
disp_lda = ConfusionMatrixDisplay(confusion_matrix=lda_cm, display_labels=np.arange(10))
disp_lda.plot(ax=axes[0, 1], cmap='Blues', values_format='d')
axes[0, 1].set_title(f'LDA + Gaussian\nAccuracy: {lda_accuracy*100:.2f}%')

# SVM (PCA) Confusion Matrix
disp_svm_pca = ConfusionMatrixDisplay(confusion_matrix=svm_pca_cm, display_labels=np.arange(10))
disp_svm_pca.plot(ax=axes[1, 0], cmap='Blues', values_format='d')
axes[1, 0].set_title(f'SVM (PCA reduced)\nAccuracy: {svm_pca_accuracy*100:.2f}%')

# SVM (LDA) Confusion Matrix
disp_svm_lda = ConfusionMatrixDisplay(confusion_matrix=svm_lda_cm, display_labels=np.arange(10))
disp_svm_lda.plot(ax=axes[1, 1], cmap='Blues', values_format='d')
axes[1, 1].set_title(f'SVM (LDA reduced)\nAccuracy: {svm_lda_accuracy*100:.2f}%')

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print(f"PCA + Gaussian:  {pca_accuracy*100:.2f}%")
print(f"LDA + Gaussian:  {lda_accuracy*100:.2f}%")
print(f"SVM (PCA):       {svm_pca_accuracy*100:.2f}%")
print(f"SVM (LDA):       {svm_lda_accuracy*100:.2f}%")



