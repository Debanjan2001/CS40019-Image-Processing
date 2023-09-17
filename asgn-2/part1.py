"""
Name - Debanjan Saha
Roll - 19CS30014
Assignment - 2
Part-1: Thinning algorithm
"""
import cv2
import numpy as np
# import matplotlib.pyplot as plt


class LibImgThinner:
    """
    End to End class for applying Thinning algorithm with otsu thresholding
    """
    
    def __init__(self):
        # 8-Neighbourhood in the clockwise order
        self.dx = [-1,-1, 0, +1, +1, +1, 0, -1]
        self.dy = [0, +1, +1, +1, 0, -1, -1, -1]

    def read_image(self, img_path):
        self.image_path = img_path
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self.gray_image

    def apply_otsu_thresholding(self, image):
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        total = image.size
        prob = hist / total

        max_variance = 0
        optimal_threshold = 0
        tot_sum = np.sum(np.arange(256) * prob)

        probability = [0, 0]
        sum_val = [0, 0]
        mean = [0, 0]
        for t in range(256):
            probability[0] = np.sum(prob[:t+1])
            probability[1] = 1 - probability[0]

            if probability[0] == 0 or probability[1] == 0:
                continue

            sum_val[0] = np.sum(np.arange(t+1) * prob[:t+1])
            sum_val[1] = tot_sum - sum_val[0]

            for i in range(2):
                mean[i] =  sum_val[i] / probability[i]

            bw_class_variance = probability[0] * probability[1] * (mean[0] - mean[1])**2

            if bw_class_variance > max_variance:
                max_variance = bw_class_variance
                optimal_threshold = t

        binary_image = (image > optimal_threshold).astype(np.uint8) * 255
        return binary_image
    
    def make_255_as_1(self, image):
        image = image // 255
        return image
   
    def apply_thinning(self, image):
        image = image.copy() # Do not change the original image
        image = self.make_255_as_1(image)
        image = (self.skeletonize(image) * 255).astype('uint8')
        return image

    def get_neighbours(self, x, y, image):
        "Return 8-neighbours of image point P1(x,y), in a clockwise order"
        output = [
            image[x + self.dx[i], y + self.dy[i]]
            for i in range(len(self.dx))
        ]
        return output

    def count_transitions(self, neighbours):
        # P2, P3,P8, P9, P2
        n = neighbours
        n.append(neighbours[0])
        # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)
        paired_list = zip(n, n[1:])
         # count the transitions from 0 -> 1
        return sum((n1, n2) == (0, 1) for n1, n2 in paired_list)

    def skeletonize(self, image):
        skeleton = image.copy()
        skeleton = np.logical_xor(np.ones_like(image), skeleton)
        height, width = skeleton.shape
        max_iter = 25
        itr = 0
        print(f"Please wait a few seconds for {max_iter} iterations to complete...")
        while itr < max_iter:
            itr += 1
            print(f"Iteration: {itr}")
            changed = False
            marked_for_deletion = []
            for x in range(1, height - 1):
                for y in range(1, width - 1):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = self.get_neighbours(x, y, skeleton)
                    """ 
                    Condition 1: Point P1 in the object regions
                    Condition 2: 2<=N(P1)<=6
                    Condition 3: S(P1)=1
                    Condition 4: P2 * P4 * P6 = 0
                    Condition 5: P4 * P6 * P8 = 0
                    """
                    if (skeleton[x,y] == 1 
                        and 2 <= sum(n) <= 6 
                        and self.count_transitions(n) == 1 
                        and P2 * P4 * P6 == 0 
                        and P4 * P6 * P8 == 0
                    ):
                        marked_for_deletion.append((x,y))

            for x, y in marked_for_deletion:
                if not changed:
                    changed = True
                skeleton[x, y] = 0

            if not changed:
                break
       
        print("Done...")
        return skeleton

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")
        return

def main():
    image_path = 'thin.jpg'
    image_thinner = LibImgThinner()
    gray_image = image_thinner.read_image(image_path)
    binary_thresholded_image = image_thinner.apply_otsu_thresholding(gray_image)
    thinned_image = image_thinner.apply_thinning(binary_thresholded_image)
    image_thinner.save_image(thinned_image, 'thinned_image.jpg')

if __name__ == '__main__':
    main()