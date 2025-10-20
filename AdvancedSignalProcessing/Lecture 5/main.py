### Task 1.1 ###

import numpy as np
import matplotlib.pyplot as plt

def LMMSE(X_means, X, C_XX, C_Theta_X):
    C_XX_inv = np.linalg.inv(C_XX)
    return np.round(X_means[3]+np.dot(np.dot(C_Theta_X,C_XX_inv),(X-X_means[:3])),4)

def MSE(var_theta, C_Theta_X, C_XX):
    C_XX_inv = np.linalg.inv(C_XX)
    C_X_Theta = np.transpose(C_Theta_X)
    return np.round(var_theta-(np.dot(np.dot(C_Theta_X, C_XX_inv),C_X_Theta)), 4)

def estimate_multiple_realizations(means, C_X, num_realizations=100):
    C_XX = C_X[:3, :3]
    C_THETA_X = C_X[3, :3]
    
    true_x4 = []
    estimated_x4 = []
    estimation_errors = []
    
    for i in range(num_realizations):
        X_full = np.random.multivariate_normal(means, C_X, 1).flatten() # Draw correlated Gaussian vector using Kay's method
        X = X_full[:3]
        true_x4_val = X_full[3]
        
        estimated_x4_val = LMMSE(means, X, C_XX, C_THETA_X)
        error = true_x4_val - estimated_x4_val
        
        true_x4.append(true_x4_val)
        estimated_x4.append(estimated_x4_val)
        estimation_errors.append(error)
    
    return {
        'true_x4': np.array(true_x4),
        'estimated_x4': np.array(estimated_x4),
        'errors': np.array(estimation_errors)
    }

def plot_correlated_gaussian(results):
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.scatter(results['true_x4'], results['estimated_x4'], alpha=0.6)
    plt.plot([min(results['true_x4']), max(results['true_x4'])], 
                [min(results['true_x4']), max(results['true_x4'])], 'r--', label='Perfect estimation')
    plt.xlabel('True X4')
    plt.ylabel('Estimated X4')
    plt.title('LMMSE Estimation Performance: True vs Estimated X4')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.hist(results['true_x4'], bins=30, alpha=0.7, density=True, label='True X4', color='blue')
    plt.hist(results['estimated_x4'], bins=30, alpha=0.7, density=True, label='Estimated X4', color='red')
    plt.xlabel('X4 Value')
    plt.ylabel('Density')
    plt.title('Probability Distributions: True vs Estimated X4')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.hist(results['errors'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Estimation Error (True - Estimated)')
    plt.ylabel('Frequency')
    plt.title('LMMSE Estimation Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero error')
    plt.legend()
    plt.grid(True)

    np.random.seed(0)
    G = np.array([[1, 0], [0.9, np.sqrt(1 - 0.9**2)]])
    M = 2000
    
    WZ = np.zeros((2, M))
    for m in range(M):
        x = np.random.randn()
        y = np.random.randn()
        wz = G @ np.array([x, y]) + np.array([1, 1])
        WZ[:, m] = wz
    
    W_mean_est = np.mean(WZ[0, :])
    Z_mean_est = np.mean(WZ[1, :])
    
    WZbar = np.zeros((2, M))
    WZbar[0, :] = WZ[0, :] - W_mean_est
    WZbar[1, :] = WZ[1, :] - Z_mean_est
    
    C_est = np.zeros((2, 2))
    for m in range(M):
        C_est = C_est + np.outer(WZbar[:, m], WZbar[:, m]) / M
    
    #print(f"W mean estimate: {W_mean_est:.4f}")
    #print(f"Z mean estimate: {Z_mean_est:.4f}")
    #print(f"Covariance estimate:\n{C_est}")
    
    theoretical_mean = np.array([1, 1])
    theoretical_cov = G @ G.T
    
    start_row = 2 if results is not None else 1
    
    plt.subplot(start_row, 3, 4 if results is not None else 1)
    plt.scatter(WZ[0, :], WZ[1, :], alpha=0.6, s=1)
    plt.xlabel('W')
    plt.ylabel('Z')
    plt.title("Kay's Method: Correlated Gaussian Variables W-Z")
    plt.grid(True)
    
    plt.subplot(start_row, 3, 5 if results is not None else 2)
    plt.hist(WZ[0, :], bins=50, alpha=0.7, density=True, label='W samples')
    x_range = np.linspace(WZ[0, :].min(), WZ[0, :].max(), 100)
    theoretical_pdf = (1/np.sqrt(2*np.pi*theoretical_cov[0,0])) * np.exp(-(x_range - theoretical_mean[0])**2 / (2*theoretical_cov[0,0]))
    plt.plot(x_range, theoretical_pdf, 'r-', label='Theoretical PDF')
    plt.xlabel('W')
    plt.ylabel('Density')
    plt.title('Variable W: Sample vs Theoretical Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(start_row, 3, 6 if results is not None else 3)
    plt.hist(WZ[1, :], bins=50, alpha=0.7, density=True, label='Z samples')
    z_range = np.linspace(WZ[1, :].min(), WZ[1, :].max(), 100)
    theoretical_pdf_z = (1/np.sqrt(2*np.pi*theoretical_cov[1,1])) * np.exp(-(z_range - theoretical_mean[1])**2 / (2*theoretical_cov[1,1]))
    plt.plot(z_range, theoretical_pdf_z, 'r-', label='Theoretical PDF')
    plt.xlabel('Z')
    plt.ylabel('Density')
    plt.title('Variable Z: Sample vs Theoretical Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return WZ, W_mean_est, Z_mean_est, C_est

def main():
    print("########################## Single realisation ##########################")
    # I am given X = [X_1, X_2, X_3, X_4]. I want to estimate X_4 given X_1 X_2 X_3
    # Therefore i now want to estimate Theta = X4 and X = [X_1, X_2, X-3]    
    
    C_X = np.array([[1, -1, 0.5, -1], #Given from task
                    [-1, 5, 2.5, 3],
                    [0.5, 2.5, 6.5, 2],
                    [-1, 3, 2, 2.5]])
    
    C_XX = np.array([[1, -1, 0.5], #Use subset of matrix for new defined X
                    [-1, 5, 2.5,],
                    [0.5, 2.5, 6.5]])

    C_THETA_X = C_X[3][:3] # Since we want to estimate X_4
    
    means = np.array([1,-3,0,2])
    vars = np.array([1,5,6.5,2.5])

    X = np.array([0.3,-1,4]) #Given from task

    theta_hat = LMMSE(X_means=means, X=X, C_XX=C_XX, C_Theta_X=C_THETA_X)
    print(f"Theta_hat: {theta_hat}")

    mse = MSE(var_theta=vars[3], C_Theta_X=C_THETA_X, C_XX=C_XX)
    print(f"MSE: {mse}")

    print("########################## Multiple realisations ##########################")
    
    results = estimate_multiple_realizations(means, C_X, num_realizations=10000)
    
    empirical_mse = np.mean(results['errors']**2)
    theoretical_mse = MSE(vars[3], C_THETA_X, C_XX)
    
    print(f"Empirical MSE: {empirical_mse:.4f}")
    print(f"Theoretical MSE: {theoretical_mse:.4f}")
    print(f"Mean estimation error: {np.mean(results['errors']):.4f}")
    
    plot_correlated_gaussian(results)

if __name__ == "__main__":
    main()
