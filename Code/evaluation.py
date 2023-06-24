import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from PIL import Image





def calculate_metrics(original_image, compressed_image):
    original_array = np.array(original_image)
    compressed_array = np.array(compressed_image)
    
    psnr_value = psnr(original_array, compressed_array)
    mse_value = mse(original_array, compressed_array)
    ssim_value = ssim(original_array, compressed_array, multichannel=True)
    
    return psnr_value, mse_value, ssim_value

def plot_metrics(psnr_values, mse_values, ssim_values, labels):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(labels, psnr_values, marker='o')
    plt.xlabel('Compression Technique')
    plt.ylabel('PSNR')
    plt.title('PSNR Comparison')

    plt.subplot(1, 3, 2)
    plt.plot(labels, mse_values, marker='o')
    plt.xlabel('Compression Technique')
    plt.ylabel('MSE')
    plt.title('MSE Comparison')

    plt.subplot(1, 3, 3)
    plt.plot(labels, ssim_values, marker='o')
    plt.xlabel('Compression Technique')
    plt.ylabel('SSIM')
    plt.title('SSIM Comparison')

    plt.tight_layout()
    plt.show()

# Load the original image
original_image = Image.open('image.jpeg')

# Set the number of components for PCA and number of colors for K-means
n_pca_components = 64
n_kmeans_colors = 64

# Compress the image using PCA
compressed_pca = compress_image(original_image)

# Compress the image using K-means
compressed_kmeans = compress_image_km(original_image, n_kmeans_colors)

# Calculate the metrics for both compression techniques
psnr_values = []
mse_values = []
ssim_values = []

# Calculate metrics for PCA compression
psnr_pca, mse_pca, ssim_pca = calculate_metrics(original_image, compressed_pca)
psnr_values.append(psnr_pca)
mse_values.append(mse_pca)
ssim_values.append(ssim_pca)

# Calculate metrics for K-means compression
psnr_kmeans, mse_kmeans, ssim_kmeans = calculate_metrics(original_image, compressed_kmeans)
psnr_values.append(psnr_kmeans)
mse_values.append(mse_kmeans)
ssim_values.append(ssim_kmeans)

# Create labels for the compression techniques
labels = ['PCA', 'K-means']

# Plot the performance metrics
plot_metrics(psnr_values, mse_values, ssim_values, labels)