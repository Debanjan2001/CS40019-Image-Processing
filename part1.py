import cv2
import numpy as np
from scipy.signal import gaussian
import matplotlib.pyplot as plt

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
    
    def add_gaussian_noise(self, image, variance):
        height, width = image.shape
        noise = np.random.normal(0, np.sqrt(variance), (height, width))
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    # def wiener_filter(self, noisy_image, kernel, noise_var):
    #     noisy_spectrum = np.fft.fft2(noisy_image)
    #     kernel_spectrum = np.fft.fft2(kernel, s=noisy_image.shape)
    #     filter_spectrum = np.conj(kernel_spectrum) / (np.abs(kernel_spectrum)**2 + noise_var)
    #     restored_spectrum = filter_spectrum * noisy_spectrum
    #     restored_image = np.fft.ifft2(restored_spectrum).real
    #     return np.clip(restored_image, 0, 255)

      
    def wiener_filter(self, img, kernel, K):
        kernel /= np.sum(kernel)
        dummy = np.copy(img)
        dummy = np.fft.fft2(dummy)
        kernel = np.fft.fft2(kernel, s = img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        dummy = dummy * kernel
        dummy = np.abs(np.fft.ifft2(dummy))
        return dummy

    
    def psnr(self, original, noisy):
        mse = np.mean((original - noisy)**2)
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def mse(self, original, noisy):
        return np.mean((original - noisy)**2)

    def gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp(- ((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    # def gaussian_kernel(self, kernel_size = 3):
    #     h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    #     h = np.dot(h, h.transpose())
    #     h /= np.sum(h)
    #     return h
    

def main():
    image_path = 'restore_min.jpg'
    part1Solver = Part1Processor()
    gray_image = part1Solver.read_image(image_path)

    # kernel_size = 1
    # # Apply the low-pass filter
    # blurred_image = part1Solver.apply_low_pass_filter(gray_image, kernel_size)
    # cv2.imshow('Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Add Gaussian noise
    noise_variance = 100
    noisy_image = part1Solver.add_gaussian_noise(gray_image, noise_variance)
    # cv2.imshow('Image', noisy_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ### Method:1
    # Define a simple averaging kernel 
    # kernel = np.ones((9, 9))
    # kernel /= np.sum(kernel)

    # Assuming we know the variance of the noise (which is 100 in this case)
    # noise_var = 100

    # Apply Wiener filter
    # restored_image = part1Solver.wiener_filter(noisy_image, kernel, noise_variance)
    # cv2.imshow('Image', restored_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ### Method:2
    kernel = part1Solver.gaussian_kernel(3, 10)
    restored_image = part1Solver.wiener_filter(noisy_image, kernel, noise_variance)
    label = ['Original Image', 'Added Gaussian Noise', 'Wiener Filter applied']
    display = [gray_image, noisy_image, restored_image]
    fig = plt.figure(figsize=(12, 10))

    for i in range(len(display)):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])

    plt.show()
    # Calculate PSNR and MSE
    psnr_value = part1Solver.psnr(gray_image, restored_image)
    mse_value = part1Solver.mse(gray_image, restored_image)
    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'MSE: {mse_value:.2f}')

if __name__ == '__main__':
    main()