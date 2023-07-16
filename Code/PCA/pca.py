# %%
import numpy as np
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

def calculate_mse(image1, image2):
    # Calculate the mean squared error
    mse = np.mean((image1 - image2) ** 2)
    return mse

# Load the image
image_raw = imread("../test.jpg")

# Perform PCA
pca = PCA()
pca.fit(image_raw)

variance = np.cumsum(pca.explained_variance_ratio_) * 100

# Calculating the number of components needed to preserve 98% of the data
k = np.argmax(variance > 98)

# Perform Incremental PCA for compression
ipca = IncrementalPCA(n_components=k)
image_compressed = ipca.inverse_transform(ipca.fit_transform(image_raw))

imsave('./pca_ghyb.jpg', image_compressed)

# Calculate compression ratio
original_size = os.stat('../test.jpg').st_size
compressed_size = os.stat('./pca_ghyb.jpg').st_size
compression_ratio = original_size / compressed_size

# Calculate MSE
mse = calculate_mse(image_raw, image_compressed)

# Calculate PSNR
psnr = peak_signal_noise_ratio(image_raw, image_compressed)

# Calculate SSIM
data_range = image_raw.max() - image_raw.min()
ssim = structural_similarity(image_raw, image_compressed, data_range=data_range, multichannel=True)

# Display the original and compressed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_raw, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_compressed, cmap='gray')
plt.title('Compressed Image (PCA)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print the calculated metrics
print("Compression Ratio:", compression_ratio)
print("MSE:", mse)
print("PSNR:", psnr)
print("SSIM:", ssim)



