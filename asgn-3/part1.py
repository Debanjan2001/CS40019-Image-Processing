import cv2
import numpy as np

class Part1Processor:

    def __init__(self) -> None:
        pass

    def read_image(self, img_path):
        self.image_path = img_path
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self.gray_image
    
    def apply_low_pass_filter(self, image, kernel_size):
        # Define the kernel for the low-pass filter
        height, width = image.shape
        result = np.zeros((height, width), dtype=np.uint8)
        
        offset = kernel_size // 2
        
        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                result[i, j] = np.mean(image[i-offset:i+offset+1, j-offset:j+offset+1])
        
        return result


def main():
    image_path = 'restore.jpg'
    part1Solver = Part1Processor()
    gray_image = part1Solver.read_image(image_path)

    kernel_size = 19
    # Apply the low-pass filter
    blurred_image = part1Solver.apply_low_pass_filter(gray_image, kernel_size)

    cv2.imshow('Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()