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
block_size = 6
constant = 16

# Apply custom adaptive thresholding
binary_image_custom = custom_adaptive_threshold(image, block_size, constant)

# # Display the result
# cv2.imshow('Custom Adaptive Thresholding', binary_image_custom)
# cv2.waitKey()
# cv2.destroyAllWindows()

def find_contour(binary_image):
    rows,cols = binary_image.shape
    contour = []
    for i in range(rows):
        for j in range(cols):
            if binary_image[i][j] == 0:  # Assuming 0 for foreground and 255 for background
                contour.append((i, j))
    return contour

# def calculate_distance(point1, point2):
#     sq = lambda x: x*x
#     return (sq(point1[0] - point2[0]) + sq(point1[1] - point2[1]))

# def find_next_point(current_point, contour):
#     min_distance = float('inf')
#     next_point = None
#     for point in contour:
#         distance = calculate_distance(current_point, point)
#         if distance < min_distance:
#             min_distance = distance
#             next_point = point
#     return next_point

# def minimum_perimeter_polygon(image):
#     binary_image = image.copy()
#     # binary_image[np.where((image==255))] = 0
#     # binary_image[np.where((image==0))] = 255
#     contour = find_contour(binary_image)
#     start_point = min(contour, key=lambda x: x[0] + x[1])  # Start from the point with lowest sum of coordinates

#     mpp = [start_point]
#     print(start_point, len(contour))
#     current_point = start_point

#     while len(contour) > 1:
#         print(len(contour))
#         contour.remove(current_point)
#         next_point = find_next_point(current_point, contour)
#         if next_point == start_point:
#             break
#         mpp.append(next_point)
#         current_point = next_point

#     return mpp


# plt.imshow(binary_image_custom, cmap='gray')
plt.imsave('poly_image.jpg', binary_image_custom, cmap='gray')
# plt.show()


# print(binary_image_custom)

def draw_mpp(image, mpp):
    mpp = np.array(mpp)
    cv2.polylines(image, [mpp], isClosed=True, color=(0, 255, 0), thickness=1)

# mpp = minimum_perimeter_polygon(binary_image_custom)

# Draw MPP on the image
# draw_mpp(image, mpp)

# output_path = 'poly_image.jpg'
# cv2.imwrite(output_path, image)

def create_polygon(points):
    """Finds the minimum perimeter polygon that encloses a given set of points.

    Args:
        points: A list of (x, y) coordinates.

    Returns:
        A list of (x, y) coordinates representing the vertices of the polygon.
    """
    hull = []
    points.sort(key=lambda x:[x[0],x[1]])
    start = points.pop(0)

    def get_slope(p1, p2):
        if p1[0] == p2[0]:
            return float('inf')
        else:
            return 1.0*(p1[1]-p2[1])/(p1[0]-p2[0])

    points.sort(key=lambda p: (get_slope(p, start), -p[1],p[0]))


    def get_cross_product(p1,p2,p3):
        return ((p2[0] - p1[0])*(p3[1] - p1[1])) - ((p2[1] - p1[1])*(p3[0] - p1[0]))

    hull.append(start)
    for p in points:
        hull.append(p)
        while len(hull) > 2 and get_cross_product(hull[-3],hull[-2],hull[-1]) < 0:
            hull.pop(-2)

    return hull


def find_hull2(points):
    def leftmost(points):
        minim = 0
        for i in range(1,len(points)):
            if points[i][0] < points[minim][0]:
                minim = i
            elif points[i][0] == points[minim][0]:
                if points[i][1] > points[minim][1]:
                    minim = i
        return minim
    
    def det(p1, p2, p3):
        """ 
        > 0: CCW turn
        < 0 CW turn
        = 0: colinear
        """
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) \
            -(p2[1] - p1[1]) * (p3[0] - p1[0])
    
    hull = []
    l = leftmost(points)
    leftMost = points[l]
    currentVertex = leftMost
    hull.append(currentVertex)
    nextVertex = points[1]
    index = 2
    nextIndex = -1
    while True:
        c0 = currentVertex
        c1 = nextVertex

        checking = points[index]
        c2 = checking

        crossProduct = det(currentVertex, nextVertex, checking)
        if crossProduct < 0:
            nextVertex = checking
            nextIndex = index
        index += 1
        if index == len(points):
            if nextVertex == leftMost:
                break
            index = 0
            hull.append(nextVertex)
            currentVertex = nextVertex
            nextVertex = leftMost

    return hull

contours = find_contour(binary_image_custom)

hull = create_polygon(contours)
# hull = find_hull2(contours)
print(hull)

temp_image = np.ones_like(image) * 255
temp_image = image.copy()
# for pt in contours:
    # temp_image[pt[0], pt[1]] = 0

hull.reverse()

# for point in hull:
#     cv2.circle(temp_image, point, 1, (0, 0), -1)

# np.rot90(temp_image, 3)
draw_mpp(temp_image, hull)

# # Display the image
cv2.imshow('Contour Points', temp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('poly_image.jpg', temp_image)
