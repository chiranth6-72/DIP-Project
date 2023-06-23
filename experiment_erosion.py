# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np
  
# Reading the input image
img = cv2.imread('lena.tiff')
  
# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
threshold_lower = 150  # Lower Threshold
threshold_upper = 250  # Upper threshold
edges = cv2.Canny(img, threshold_lower, threshold_upper)
cv2.imshow('edge image',edges)
# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img, kernel, iterations=2)
img_subtracted=cv2.subtract(img,img_erosion)
cv2.imshow('subtracted image',img_subtracted)
cv2.imwrite('subtracted image.png',img_subtracted)
img_dilation = cv2.dilate(img, kernel, iterations=2) 
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.imwrite('Input.png', img)
cv2.imwrite('Erosion.png', img_erosion)
cv2.imwrite('Dilation.png', img_dilation)
erosion_edges = cv2.erode(edges, kernel, iterations = 2)
# Apply dilation to the edge image
dilation_edge = cv2.dilate(edges, kernel, iterations = 2)
#cv2.imshow("Original Image", img)
cv2.imshow("Edge Image using Erosion", erosion_edges)
cv2.imshow("Edge Image using Dilation", dilation_edge)
cv2.imwrite('Edge Image using Dilation.png', dilation_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
