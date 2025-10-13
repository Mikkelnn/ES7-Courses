import scipy.io as io
import numpy as np
import pathlib as path
import matplotlib.pyplot as plt 

file = path.Path("./RawMRI.mat")

mat = io.loadmat(file)


def inverse_fft2_shift(kspace):
    return (np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2,-1)), norm='ortho'))


def square_high_pass_filter(data, interval):
    sizex, sizey = data.shape
    
    # Create a copy to avoid modifying original data
    filtered_data = np.copy(data)

    # Find centers
    center_x = sizex // 2
    center_y = sizey // 2

    # Calculate start and end indices, ensuring they are within bounds
    start_x = max(center_x - interval, 0)
    end_x = min(center_x + interval, sizex)
    start_y = max(center_y - interval, 0)
    end_y = min(center_y + interval, sizey)

    # Set the interval around the center to zero
    filtered_data[start_x:end_x, start_y:end_y] = 0
    return filtered_data

def square_low_pass_filter(data, interval):
    sizex, sizey = data.shape
    
    # Create a copy to avoid modifying original data
    filtered_data = np.copy(data)
    
    # Find centers
    center_x = sizex // 2
    center_y = sizey // 2

    # Calculate start and end indices, ensuring they are within bounds
    start_x = max(center_x - interval, 0)
    end_x = min(center_x + interval, sizex)
    start_y = max(center_y - interval, 0)
    end_y = min(center_y + interval, sizey)

    # Keep only the interval around the center (set everything else to zero)
    mask = np.zeros_like(filtered_data)
    mask[start_x:end_x, start_y:end_y] = 1
    filtered_data = filtered_data * mask
    
    return filtered_data

def circular_high_pass_filter(data, radius):
    sizex, sizey = data.shape
    
    # Create a copy to avoid modifying original data
    filtered_data = np.copy(data)
    
    # Find centers
    center_x = sizex // 2
    center_y = sizey // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:sizex, :sizey]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create circular mask (set frequencies within radius to zero)
    mask = distance > radius
    filtered_data = filtered_data * mask
    
    return filtered_data

def circular_low_pass_filter(data, radius):
    sizex, sizey = data.shape
    
    # Create a copy to avoid modifying original data
    filtered_data = np.copy(data)
    
    # Find centers
    center_x = sizex // 2
    center_y = sizey // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:sizex, :sizey]
    
    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create circular mask (keep only frequencies within radius)
    mask = distance <= radius
    filtered_data = filtered_data * mask
    
    return filtered_data

def main():
    brain1 = np.array(mat["fbrain1"])

    # Filter parameters
    square_interval = 20  # For square filters
    circular_radius = 20  # For circular filters

    # Apply square filters
    brain1_square_low = square_low_pass_filter(brain1, square_interval)
    brain1_square_high = square_high_pass_filter(brain1, square_interval)
    
    # Apply circular filters
    brain1_circular_low = circular_low_pass_filter(brain1, circular_radius)
    brain1_circular_high = circular_high_pass_filter(brain1, circular_radius)

    # Extract real and imaginary parts of original k-space
    brain1_img = np.imag(brain1)
    brain1_real = np.real(brain1)

    # Compute inverse FFTs for all filtered data
    fft_original = inverse_fft2_shift(brain1)
    fft_square_high = inverse_fft2_shift(brain1_square_high)
    fft_square_low = inverse_fft2_shift(brain1_square_low)
    fft_circular_high = inverse_fft2_shift(brain1_circular_high)
    fft_circular_low = inverse_fft2_shift(brain1_circular_low)

    # Create comprehensive subplot layout
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    
    # Row 1: K-space visualization
    axs[0, 0].imshow(np.log(np.abs(brain1_real) + 1e-10), cmap='gray')
    axs[0, 0].set_title('K-space Real Part\n(log scale)')
    axs[0, 0].set_xlabel('X-axis')
    axs[0, 0].set_ylabel('Y-axis')

    axs[0, 1].imshow(np.log(np.abs(brain1_img) + 1e-10), cmap='gray')
    axs[0, 1].set_title('K-space Imaginary Part\n(log scale)')
    axs[0, 1].set_xlabel('X-axis')
    axs[0, 1].set_ylabel('Y-axis')

    axs[0, 2].imshow(np.log(np.abs(brain1) + 1e-10), cmap='gray')
    axs[0, 2].set_title('K-space Magnitude\n(log scale)')
    axs[0, 2].set_xlabel('X-axis')
    axs[0, 2].set_ylabel('Y-axis')

    # Row 2: Square filter results in K-space
    axs[1, 0].imshow(np.log(np.abs(brain1) + 1e-10), cmap='gray')
    axs[1, 0].set_title('Original K-space\n(log scale)')
    axs[1, 0].set_xlabel('X-axis')
    axs[1, 0].set_ylabel('Y-axis')

    axs[1, 1].imshow(np.log(np.abs(brain1_square_high) + 1e-10), cmap='gray')
    axs[1, 1].set_title('Square High-Pass\nFiltered K-space')
    axs[1, 1].set_xlabel('X-axis')
    axs[1, 1].set_ylabel('Y-axis')

    axs[1, 2].imshow(np.log(np.abs(brain1_square_low) + 1e-10), cmap='gray')
    axs[1, 2].set_title('Square Low-Pass\nFiltered K-space')
    axs[1, 2].set_xlabel('X-axis')
    axs[1, 2].set_ylabel('Y-axis')

    # Row 3: Circular filter results in K-space
    axs[2, 0].imshow(np.log(np.abs(brain1) + 1e-10), cmap='gray')
    axs[2, 0].set_title('Original K-space\n(Reference)')
    axs[2, 0].set_xlabel('X-axis')
    axs[2, 0].set_ylabel('Y-axis')

    axs[2, 1].imshow(np.log(np.abs(brain1_circular_high) + 1e-10), cmap='gray')
    axs[2, 1].set_title('Circular High-Pass\nFiltered K-space')
    axs[2, 1].set_xlabel('X-axis')
    axs[2, 1].set_ylabel('Y-axis')

    axs[2, 2].imshow(np.log(np.abs(brain1_circular_low) + 1e-10), cmap='gray')
    axs[2, 2].set_title('Circular Low-Pass\nFiltered K-space')
    axs[2, 2].set_xlabel('X-axis')
    axs[2, 2].set_ylabel('Y-axis')

    # Add overall title and adjust layout
    fig.suptitle('MRI Signal Processing: K-space Analysis and Spatial Filtering Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()
    
    # Additional plot: Filter comparison side by side
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # High-pass comparison
    axs2[0, 0].imshow(np.abs(fft_original), cmap='gray')
    axs2[0, 0].set_title('Original')
    axs2[0, 0].set_xlabel('X-axis')
    axs2[0, 0].set_ylabel('Y-axis')
    
    axs2[0, 1].imshow(np.abs(fft_square_high), cmap='gray')
    axs2[0, 1].set_title('Square High-Pass')
    axs2[0, 1].set_xlabel('X-axis')
    axs2[0, 1].set_ylabel('Y-axis')
    
    axs2[0, 2].imshow(np.abs(fft_circular_high), cmap='gray')
    axs2[0, 2].set_title('Circular High-Pass')
    axs2[0, 2].set_xlabel('X-axis')
    axs2[0, 2].set_ylabel('Y-axis')
    
    # Low-pass comparison
    axs2[1, 0].imshow(np.abs(fft_original), cmap='gray')
    axs2[1, 0].set_title('Original')
    axs2[1, 0].set_xlabel('X-axis')
    axs2[1, 0].set_ylabel('Y-axis')
    
    axs2[1, 1].imshow(np.abs(fft_square_low), cmap='gray')
    axs2[1, 1].set_title('Square Low-Pass')
    axs2[1, 1].set_xlabel('X-axis')
    axs2[1, 1].set_ylabel('Y-axis')
    
    axs2[1, 2].imshow(np.abs(fft_circular_low), cmap='gray')
    axs2[1, 2].set_title('Circular Low-Pass')
    axs2[1, 2].set_xlabel('X-axis')
    axs2[1, 2].set_ylabel('Y-axis')
    
    fig2.suptitle('Filter Shape Comparison: Square vs Circular Filters', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
