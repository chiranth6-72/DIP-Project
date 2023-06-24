import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

def compress_image_km(image_path, k):
    # Load the image
    image = Image.open(image_path)
    
    # Convert image to numpy array
    image_array = np.array(image, dtype=np.float64) / 255
    
    # Reshape the array to 2D (pixels as feature vectors)
    rows, cols, channels = image_array.shape
    image_2d = image_array.reshape(rows * cols, channels)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image_2d)
    
    # Get the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Replace each pixel with its assigned cluster center
    compressed_image_2d = np.array([cluster_centers[label] for label in labels])
    
    # Reshape the compressed image to its original shape
    compressed_image_array = compressed_image_2d.reshape(rows, cols, channels)
    
    # Convert the array back to PIL image
    compressed_image = Image.fromarray((compressed_image_array * 255).astype(np.uint8))
    
    compressed_image.save('compressed_image_kmeans.jpg')

    return compressed_image

# Example usage
# image_path = 'image.jpeg'
# k = 5  # Number of clusters (colors) for compression
# compressed_image = compress_image_km(image_path, k)
# imgPath = "compressed_image_kmeans.jpg"
# print(os.stat(imgPath).st_size/1024)