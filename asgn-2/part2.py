import numpy as np
import cv2
import matplotlib.pyplot as plt

def custom_adaptive_threshold(image, block_size, constant):
    height, width = image.shape
    binary_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            mean_intensity = np.mean(block)
            threshold = mean_intensity - constant
            binary_block = (block > threshold).astype(np.uint8) * 255
            binary_image[i:i+block_size, j:j+block_size] = binary_block
    
    return binary_image

# Read the image in grayscale
image = cv2.imread('poly.jpg', cv2.IMREAD_GRAYSCALE)

# Define block size and constant (adjust as needed)
block_size = 4
constant = 2

# Apply custom adaptive thresholding
binary_image_custom = custom_adaptive_threshold(image, block_size, constant)

# # Display the result
# cv2.imshow('Custom Adaptive Thresholding', binary_image_custom)
# cv2.waitKey()
# cv2.destroyAllWindows()

# plt.imshow(binary_image_custom, cmap='gray')
plt.imsave('poly_image.jpg', binary_image_custom, cmap='gray')
# plt.show()
