import cv2
import numpy as np

class Part2Processor:

    def __init__(self) -> None:
        pass

    def read_image(self, img_path):
        self.image_path = img_path
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self.gray_image
    
    def apply_otsu_thresholding(self, image):
        pixel_number = image.shape[0] * image.shape[1]
        mean_weigth = 1.0/pixel_number
        his, bins = np.histogram(image.flatten(), bins=np.array(range(0, 256)), range=(0, 256))
        final_thresh = -1
        final_value = -1
        for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
            Wb = np.sum(his[:t]) * mean_weigth
            Wf = np.sum(his[t:]) * mean_weigth

            mub = np.mean(his[:t])
            muf = np.mean(his[t:])

            value = Wb * Wf * ((mub - muf) ** 2)

            # print("Wb", Wb, "Wf", Wf)
            # print("t", t, "value", value)

            if value >= final_value:
                final_thresh = t
                final_value = value

        final_img = image.copy()
        print(final_thresh)
        final_thresh=230
        final_img[image > final_thresh] = 255
        final_img[image < final_thresh] = 0
        return final_img
    
    def otsu2(self, image):
        # Compute histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0,256))

        # Normalize histogram
        hist = hist / hist.sum()

        # Initialization
        max_variance = 0
        threshold = 0

        for t in range(256):
            # Class probabilities
            w0 = hist[:t+1].sum()
            w1 = 1 - w0

            if w0 == 0 or w1 == 0:
                continue

            # Class means
            u0 = (np.arange(t+1) * hist[:t+1]).sum() / w0
            u1 = (np.arange(t+1, 256) * hist[t+1:]).sum() / w1

            # Class variances
            var = w0 * w1 * ((u0 - u1) ** 2)

            if abs(var - max_variance) <= 210:
                max_variance = max(max_variance, var)
                threshold = t

        # Apply threshold
        print(threshold)
        thresholded = image.copy()
        thresholded[image > threshold] = 255
        thresholded[image <= threshold] = 0
        return thresholded

    
def main():
    image_path = 'connect.png'
    part2Solver = Part2Processor()
    gray_image = part2Solver.read_image(image_path)
    print(gray_image)    

    # cv2.imshow('Real Image', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # otsu_image = part2Solver.apply_otsu_thresholding(gray_image)
    otsu_image = part2Solver.otsu2(gray_image)
    cv2.imshow('Otsu Image', otsu_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()