# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread('lena.tiff')

# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayscale image',gray)
# blur_Gaussian = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
# # Apply a Sobel filter,Laplacian filter,Canny edge detection to the blurred
#     #image to extract edges
    
# sobelx = cv2.Sobel(blur_Gaussian,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(blur_Gaussian,cv2.CV_64F,0,1,ksize=5)
# sobel_edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# laplacian = cv2.Laplacian(blur_Gaussian,cv2.CV_64F)
# canny_edges = cv2.Canny(blur_Gaussian,100,200)
# cv2.imshow('Filtered_output',blur_Gaussian)
# cv2.imshow('Sobel_edge',sobel_edges)
# cv2.imshow('laplacian',laplacian)
# cv2.imshow('Canny Edge detection',canny_edges)
# # Apply a median blur to the grayscale image
# blur_median = cv2.medianBlur(gray, 5)
# cv2.imshow('Median Blur', blur_median)
# # Apply bilateral filter with (d = Diameter of each pixel neighborhood  = 15, 
# # sigmaColor (Value of sigma  in the color space)
# # = sigmaSpace (Value of sigma  in the coordinate space)  = 75.
# bilateral = cv2.bilateralFilter(gray, 15, 75, 75) 
# # Save the output.
# cv2.imshow('bilateral Filter', bilateral)
# # Apply a Gabor filter to the grayscale image to extract textures
# ksize = 31
# sigma = 5
# theta = 2
# lamda = 10
# gamma = 1
# phi = 1
# kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda,
#                             gamma, phi, ktype=cv2.CV_32F)
# gabor = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
# cv2.imshow('Gabor Filter', gabor)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




import cv2
import numpy as np

# Read the image
img = cv2.imread('input_img.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayscale_image.jpg', gray)

# Apply a Gaussian blur to the grayscale image
blur_Gaussian = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
cv2.imwrite('blur_Gaussian.jpg', blur_Gaussian)

# Apply a Sobel filter to the blurred image to extract edges
sobelx = cv2.Sobel(blur_Gaussian, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blur_Gaussian, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imwrite('Output/sobel_edges.jpg', sobel_edges)

# Apply a Laplacian filter to the blurred image
laplacian = cv2.Laplacian(blur_Gaussian, cv2.CV_64F)
cv2.imwrite('./Output/laplacian.jpg', laplacian)

# Apply Canny edge detection to the blurred image
canny_edges = cv2.Canny(blur_Gaussian, 100, 200)
cv2.imwrite('./Output/canny_edges.jpg', canny_edges)

# Apply a median blur to the grayscale image
blur_median = cv2.medianBlur(gray, 5)
cv2.imwrite('./Output/blur_median.jpg', blur_median)

# Apply a bilateral filter to the grayscale image
bilateral = cv2.bilateralFilter(gray, 15, 75, 75)
cv2.imwrite('./Output/bilateral.jpg', bilateral)

# Apply a Gabor filter to the grayscale image to extract textures
ksize = 31
sigma = 5
theta = 2
lamda = 10
gamma = 1
phi = 1
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
gabor = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
cv2.imwrite('./Output/gabor.jpg', gabor)

# Show the images
# cv2.imshow('Grayscale Image', gray)
# cv2.imshow('Gaussian Blur', blur_Gaussian)
# cv2.imshow('Sobel Edges', sobel_edges)
# cv2.imshow('Laplacian', laplacian)
# cv2.imshow('Canny Edges', canny_edges)
# cv2.imshow('Median Blur', blur_median)
# cv2.imshow('Bilateral Filter', bilateral)
# cv2.imshow('Gabor Filter', gabor)

cv2.waitKey(0)
cv2.destroyAllWindows()
