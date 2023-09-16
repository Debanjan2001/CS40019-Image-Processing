import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image_path = 'thin.jpg'
# image_path = 'temp.png'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
def otsu_thresholding(image):
    hist, bins = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size
    prob = hist / total_pixels

    max_variance = 0
    optimal_threshold = 0
    sum_all = np.sum(np.arange(256) * prob)

    for t in range(256):
        prob_class0 = np.sum(prob[:t+1])
        prob_class1 = 1 - prob_class0

        if prob_class0 == 0 or prob_class1 == 0:
            continue

        sum_class0 = np.sum(np.arange(t+1) * prob[:t+1])
        sum_class1 = sum_all - sum_class0

        mean_class0 = sum_class0 / prob_class0
        mean_class1 = sum_class1 / prob_class1

        between_class_variance = prob_class0 * prob_class1 * (mean_class0 - mean_class1)**2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = t

    binary_image = (image > optimal_threshold).astype(np.uint8) * 255

    return binary_image, optimal_threshold

binary_image, _ = otsu_thresholding(gray_image)

def zhang_suen_thinning(image):
    print(image)
    image  = image // 255
    def neighbours(x,y,image):
        "Return 8-neighbours of image point P1(x,y), in a clockwise order"
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        return [ img[x_1,y], img[x_1,y1], img[x,y1], img[x1,y1],     # P2,P3,P4,P5
                img[x1,y], img[x1,y_1], img[x,y_1], img[x_1,y_1] ]    # P6,P7,P8,P9

    def transitions(neighbours):
        "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
        n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
        return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

    def zhangSuen(image):
        "the Zhang-Suen Thinning Algorithm"
        skeleton = image.copy()  # deepcopy to protect the original image
        skeleton[np.where((image==1))] = 0
        skeleton[np.where((image==0))] = 1
        # plt.imshow(skeleton)
        rows, columns = skeleton.shape
        has_changed = True
        itr = 0
        while has_changed and itr < 10:
            has_changed = False
            itr += 1
            print(has_changed)
            marked_cells = []
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x,y,skeleton)
                    if (skeleton[x,y] == 1 and    # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and   # Condition 1: 2<=N(P1)<=6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):   # Condition 4
                        marked_cells.append((x,y))
                        has_changed = True
            for x, y in marked_cells:
                skeleton[x, y] = 0
        return skeleton

    return (zhangSuen(image) * 255).astype('uint8')


thinned_image = zhang_suen_thinning(binary_image)

# Display the results
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')

plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Otsu Thresholded Image')

plt.subplot(1, 3, 3)
plt.imshow(thinned_image, cmap='gray')
plt.title('Thinned Image')

plt.show()
plt.imsave('thinned_image.jpg', thinned_image, cmap='gray')

# Save the thinned image
# cv2.imwrite('thinned_image.jpg', thinned_image)
