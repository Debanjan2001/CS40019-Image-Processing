import cv2
import numpy as np
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
        print("[Started] Blurring image...")
        height, width = image.shape
        result = np.zeros((height, width), dtype=np.uint8)
        offset = kernel_size // 2
        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                result[i, j] = np.mean(image[i-offset:i+offset+1, j-offset:j+offset+1])
        print("[Finished] Blurring image.")
        return result
    
    def add_gaussian_noise(self, image, variance):
        print("[Started] Adding noise to image...")
        height, width = image.shape
        noise = np.random.normal(0, np.sqrt(variance), (height, width))
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        print("[Finished] Adding noise to image.")
        return noisy_image
    
    def normalize(self, img):
        image = img.copy()
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        return image.astype(np.uint8)
    
    def wiener_filter(self, img, kernel, var):
        print("[Started] Applying Wiener filter to image...")
        dummy = np.copy(img)
        dummy = np.fft.fft2(dummy)
        kernel = np.fft.fft2(kernel, s = img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + var)
        dummy = dummy * kernel
        dummy = np.abs(np.fft.ifft2(dummy))
        dummy = self.normalize(dummy)
        print("[Finished] Applying Wiener filter to image.")
        return dummy

    def psnr(self, original, noisy):
        mse = self.mse(original, noisy)
        return 10 * np.log10(255**2 / mse)

    def mse(self, original, noisy):
        return np.mean((original - noisy)**2)

    def gaussian_kernel(self, kernel_size, sigma):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        offset = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i-offset)**2 + (j-offset)**2) / (2*sigma**2))
        return kernel / np.sum(kernel)


def main():
    image_path = 'restore.jpg'
    part1Solver = Part1Processor()
    gray_image = part1Solver.read_image(image_path)

    kernel_size = 10
    # Apply the low-pass filter
    blurred_image = part1Solver.apply_low_pass_filter(gray_image, kernel_size)
    plt.imsave('part1_blurred_image.jpg', blurred_image, cmap='gray')

    # Add Gaussian noise
    noise_variance = 100
    noisy_image = part1Solver.add_gaussian_noise(blurred_image, noise_variance)
    plt.imsave('part1_noisy_image.jpg', noisy_image, cmap='gray')

    kernel = part1Solver.gaussian_kernel(4, 10)
    restored_image = part1Solver.wiener_filter(noisy_image, kernel, noise_variance)
    plt.imsave('part1_restored_image.jpg', restored_image, cmap='gray')
    
    # Calculate PSNR and MSE
    psnr_value = part1Solver.psnr(gray_image, restored_image)
    mse_value = part1Solver.mse(gray_image, restored_image)
    print(f'PSNR: {psnr_value:.2f} dB')
    print(f'MSE: {mse_value:.2f}')

if __name__ == '__main__':
    main()