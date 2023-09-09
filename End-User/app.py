from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from sklearn.decomposition import PCA, IncrementalPCA
from skimage import color

app = Flask(__name__)
  

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        img_path = 'static/uploads/uploaded_image.jpg'
        file.save(img_path)
        return compress_image(img_path)
        # return compress_image2(img_path)
    return render_template('index.html')

def compress_image(img_path):
    # Perform image compression using the provided code
    
    # Read the image
    orig_img = Image.open(img_path)
    
    orig_img = orig_img.resize((1024, 1024))
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
        
        return compressed_image

    orig_img_data, compressed_img_data = {}, {}

    # Orig img data
    img_size = os.stat(img_path).st_size/1024
    data = orig_img.getdata()
    og_pxl = np.array(data).reshape(*orig_img.size, -1)
    img_dim = og_pxl.shape
    
    orig_img_data['img_size_kb'] = img_size
    orig_img_data['img_dim'] = img_dim


    pca_channel = pca_compose(orig_img)
    
    
    compressed_image = pca_transform(pca_channel)
    
    
    
    
    # Save the compressed image
    compressed_img_path = 'static/uploads/compressed_image.jpg'
    Image.fromarray(compressed_image).save(compressed_img_path)
    
    comp_img = Image.open(compressed_img_path)
    
    # Compressed img data
    img_size_comp = os.stat(compressed_img_path).st_size/1024
    data_comp = comp_img.getdata()
    comp_pxl = np.array(data_comp).reshape(*comp_img.size, -1)
    img_dim_comp = comp_pxl.shape
    
    compressed_img_data['img_size_kb'] = img_size_comp
    compressed_img_data['img_dim'] = img_dim_comp
    
    # Display the original and compressed images
    # return render_template('result.html', orig_img_path=img_path, compressed_img_path=compressed_img_path, )

    return render_template('result.html', orig_img_path=img_path, orig_img_dim=orig_img_data['img_dim'],
                           orig_img_size=orig_img_data['img_size_kb'], compressed_img_path=compressed_img_path,
                           compressed_img_dim=compressed_img_data['img_dim'],
                           compressed_img_size=compressed_img_data['img_size_kb'])


def compress_image2(img_path):
    img = color.rgb2gray(imread(img_path))
    
    pca = PCA()
    pca.fit(img)

    variance = np.cumsum(pca.explained_variance_ratio_)*100
    k = np.argmax(variance>98)
    
    ipca = IncrementalPCA(n_components=k)
    image_compressed = ipca.inverse_transform(ipca.fit_transform(img))

    imsave('static/uploads/compressed_image_km.jpg', image_compressed)
    
    image_compressed = Image.open('static/uploads/compressed_image_km.jpg')
    

    orig_img_data, compressed_img_data = {}, {}

    orig_img = Image.open(img_path)
    # Orig img data
    img_size = os.stat(img_path).st_size/1024
    data = orig_img.getdata()
    og_pxl = np.array(data).reshape(*orig_img.size, -1)
    img_dim = og_pxl.shape
    
    orig_img_data['img_size_kb'] = img_size
    orig_img_data['img_dim'] = img_dim


    # pca_channel = pca_compose(orig_img)
    
    
    # compressed_image = pca_transform(pca_channel)
    
    
    
    
    # Save the compressed image
    compressed_img_path_pca = 'static/uploads/compressed_image_km.jpg'
    # Image.fromarray(image_compressed).save(compressed_img_path_pca)
    
    comp_img = Image.open(compressed_img_path_pca)
    
    # Compressed img data
    img_size_comp = os.stat(compressed_img_path_pca).st_size/1024
    data_comp = comp_img.getdata()
    comp_pxl = np.array(data_comp).reshape(*comp_img.size, -1)
    img_dim_comp = comp_pxl.shape
    
    compressed_img_data_pca['img_size_kb'] = img_size_comp
    compressed_img_data_pca['img_dim'] = img_dim_comp
    
    # Display the original and compressed images
    # return render_template('result.html', orig_img_path=img_path, compressed_img_path=compressed_img_path, )

    return render_template('result.html', orig_img_path=img_path, orig_img_dim=orig_img_data['img_dim'],
                           orig_img_size=orig_img_data['img_size_kb'], compressed_img_path=compressed_img_path_pca,
                           compressed_img_dim=compressed_img_data_pca['img_dim'],
                           compressed_img_size=compressed_img_data_pca['img_size_kb'])



if __name__ == '__main__':
    app.run()
