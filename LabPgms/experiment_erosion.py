import cv2
import numpy as np

# Reading the input image
img = cv2.imread('./input_img.jpg')

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
threshold_lower = 150  # Lower Threshold
threshold_upper = 250  # Upper threshold

# Apply Canny edge detection
edges = cv2.Canny(img, threshold_lower, threshold_upper)
# cv2.imshow('Edge Image', edges)
cv2.imwrite('./Output2/edge_image.png', edges)

# Erosion
img_erosion = cv2.erode(img, kernel, iterations=2)
img_subtracted = cv2.subtract(img, img_erosion)
# cv2.imshow('Subtracted Image', img_subtracted)
cv2.imwrite('./Output2/subtracted_image.png', img_subtracted)

# Dilation
img_dilation = cv2.dilate(img, kernel, iterations=2)
# cv2.imshow('Input', img)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Dilation', img_dilation)
cv2.imwrite('./Output2/input.png', img)
cv2.imwrite('./Output2/erosion.png', img_erosion)
cv2.imwrite('./Output2/dilation.png', img_dilation)

# Erosion and Dilation on edge image
erosion_edges = cv2.erode(edges, kernel, iterations=2)
dilation_edges = cv2.dilate(edges, kernel, iterations=2)
# cv2.imshow('Edge Image using Erosion', erosion_edges)
# cv2.imshow('Edge Image using Dilation', dilation_edges)
cv2.imwrite('./Output2/edge_image_erosion.png', erosion_edges)
cv2.imwrite('./Output2/edge_image_dilation.png', dilation_edges)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
