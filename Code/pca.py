from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def compress_image(img_path):
    # Perform image compression using the provided code
    
    # Read the image
    orig_img = Image.open(img_path)
    
    # Perform PCA compression
    # ...
    # Insert the code for performing PCA compression here
    # ...
    
    def pca_compose(orig_img):
        # 2. Convert the reading into a 2D numpy array
        img = np.array(orig_img.getdata())
        
        # 3. Reshape 2D to 3D array 
        # The asterisk (*) operator helps in unpacking the sequence/collection as positional arguments. 
        # So, instead of using indices of elements separately, we can use * and perform action on it.
        # print(orig_img.size) = (1024, 1024) --> print(*orig_img.size) = 1024 1024
        img = img.reshape(*orig_img.size, -1)
        
        # Separate channels from image and use PCA on each channel
        pca_channel = {}
        img_t = np.transpose(img) # transposing the image 
        
        for i in range(img.shape[-1]):    # For each RGB channel compute the PCA
            
            per_channel = img_t[i] # It will be in a shape (1,1024,1024)
            
            # Converting (1, 1024, 1024) to (1024, 1024)
            channel = img_t[i].reshape(*img.shape[:-1])  # obtain channel
            
            pca = PCA(random_state = 42)                #initialize PCA
            
            fit_pca = pca.fit_transform(channel)        #fit PCA
            
            pca_channel[i] = (pca,fit_pca)  #save PCA models for each channel
            
        return pca_channel
    
    
    def pca_transform(pca_channel, n_components=50):
    
        temp_res = []
        
        # Looping over all the channels we created from pca_compose function
        
        for channel in range(len(pca_channel)):
            
            pca, fit_pca = pca_channel[channel]
            
            # Selecting image pixels across first n components
            pca_pixel = fit_pca[:, :n_components]
            
            # First n-components
            pca_comp = pca.components_[:n_components, :]
            
            # Projecting the selected pixels along the desired n-components (De-standardization)
            compressed_pixel = np.dot(pca_pixel, pca_comp) + pca.mean_
            
            # Stacking channels corresponding to Red Green and Blue
            temp_res.append(compressed_pixel)
                
        # transforming (channel, width, height) to (height, width, channel)
        compressed_image = np.transpose(temp_res)
        
        # Forming the compressed image
        compressed_image = np.array(compressed_image,dtype=np.uint8)
        
        compressed_img_path = 'compressed_image.jpg'
        
        Image.fromarray(compressed_image).save(compressed_img_path) 
        
        return compressed_image

