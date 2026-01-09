import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import norm

x = np.array(
[
[0.7],
[1.2],
[2.2],
[2.5],
[6.5],
[7.0],
[7.4],
[7.7],
[8.3],
[8.5]
]
)
xqs = np.array([[1.9], [4.5], [7.6]])

gmm = GMM(n_components=2)
gmm.fit(x)
# print(f"Means: {gmm.means_} Variances: {gmm.covariances_}")


for i in range (0,3):
    pdf0 = np.log10(norm.pdf(xqs[i],gmm.means_[0],gmm.covariances_[0]))
    pdf1 = np.log10(norm.pdf(xqs[i],gmm.means_[1],gmm.covariances_[1]))
    print(f"xqs[{i}] log likelihood for class 0: {pdf0} log likelihood for class 1: {pdf1}")

labels = gmm.predict(xqs)
print(f"Gmm predict labels: {labels}")