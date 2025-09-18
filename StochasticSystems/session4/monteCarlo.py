import numpy as np
import matplotlib.pyplot as plt

np.random.seed(343)
#4.1 i.i.d process
n_samples = 13472
a_mean = 5
b_var = 4

bottom = a_mean - np.sqrt(3*b_var)
top = 2*a_mean - bottom

xk = np.random.normal(a_mean, np.sqrt(b_var), n_samples)
yk = np.random.uniform(bottom, top, n_samples)
λ = 0.25
zk = -1 * a_mean * np.log(top - yk)

# zk = -1 * (1 / λ) * np.log(top - yk)

# # First stem plot
# markerline1, stemlines1, baseline1 = plt.stem(xk, linefmt="C0-", markerfmt="C0o", basefmt=" ")
# plt.setp(stemlines1, color="blue")
# plt.setp(markerline1, color="blue")

# # Second stem plot
# markerline2, stemlines2, baseline2 = plt.stem(yk, linefmt="C1-", markerfmt="C1s", basefmt=" ")
# plt.setp(stemlines2, color="red")
# plt.setp(markerline2, color="red")

# # third stem plot (exponential mapping)
# markerline2, stemlines2, baseline2 = plt.stem(zk, linefmt="C1-", markerfmt="C1s", basefmt=" ")
# plt.setp(stemlines2, color="green")
# plt.setp(markerline2, color="green")

plt.hist(xk, bins=30, alpha=0.5, label="xk")
plt.hist(yk, bins=10, alpha=0.5, label="yk")
plt.hist(zk, bins=50, alpha=0.5, label="zk")


# Labels
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Stem plot of xk and yk")
plt.legend(["Normal distribution", "Uniform distribution", "zk yk transform"])
plt.show()

print(f"Mean for xk: {np.mean(xk)} Variance for xk: {np.var(xk)}")
print(f"Mean for yk: {np.mean(yk)} Variance for yk: {np.var(yk)} Bottom: {bottom} Top: {top}")
print(f"Mean for zk: {np.mean(zk)} Variance for zk: {np.var(zk)} lambda: {λ}")