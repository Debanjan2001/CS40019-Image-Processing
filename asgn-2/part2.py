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
block_size = 3
constant = 132

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

# plt.imshow(binary_image_custom, cmap='gray')
plt.imsave('poly_image.jpg', binary_image_custom, cmap='gray')
# plt.show()


def draw_mpp(image, mpp):
    mpp = np.array(mpp)
    cv2.polylines(image, [mpp], isClosed=True, color=(0, 255, 0), thickness=1)

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
            if(p1[1] > p2[1]):
                return float('inf')
            else:
                return float('-inf')
        else:
            return 1.0*(p1[1]-p2[1])/(p1[0]-p2[0])

    points.sort(key=lambda p: (p[0],p[1]))
    down_pts = []
    up_pts = []
    for pt in points:
        slope = get_slope(pt, start)
        if slope >= 0:
            up_pts.append(pt)
        else:
            down_pts.append(pt)
    
    temp_points = [start]
    temp_points.extend(down_pts)
    up_pts.sort(key=lambda p: (p[0],-p[1]), reverse=True)
    temp_points.extend(up_pts)
    # print(temp_points)
    points = temp_points
    
    # points.sort(key=lambda p: (get_slope(p, start), p[0],p[1]))

    def get_cross_product(p1,p2,p3):
        # return ((p2[0] - p1[0])*(p3[1] - p1[1])) - ((p2[1] - p1[1])*(p3[0] - p1[0]))
        return (p3[0]-p2[0])*(p1[1]-p2[1]) - (p3[1]-p2[1])*(p1[0]-p2[0])
    hull.append(start)

    isConvex = [True for i in range(len(points))]
    for i in range(len(points)):
        if get_cross_product(points[i-1], points[i], points[(i+1)%len(points)]) < 0:
            isConvex[i] = False
        else:
            isConvex[i] = True
        print(points[i], isConvex[i])

    wc=0
    bc=0
    vl=0
    vk=0
    n = len(points)
    indices = set()
    while(vk<n):
        sgn1 = get_cross_product(points[vl], points[wc], points[vk])
        sgn2 = get_cross_product(points[vl], points[bc], points[vk])
        if(sgn1 > 0):
            vl = wc
            wc = vl
            bc = vl
            vk = vl+1
        elif(sgn1 <= 0 and sgn2>=0):
            if(isConvex[vk]):
                wc = vk
            else:
                bc = vk
            vk = vk+1
        elif(sgn2<0):
            vl = bc
            wc=vl
            bc = vl
            vk = vl+1
        print(vl)
        indices.add(vl)

    print(len(indices))
    for p in points:
        hull.append(p)
        while len(hull) > 2 and get_cross_product(hull[-3],hull[-2],hull[-1]) < 0:
            hull.pop(-2)

    # return hull
    return [points[i] for i in sorted(list(indices))]


points = [[1,4],[2,3],[2,5],[3,6],[3,2],[4,9],[4,-5],[5,5], [5,9]]
# hull = create_polygon(points)
# print(hull)


# # ----------------------------
contours = find_contour(binary_image_custom)
# contours = points
hull = create_polygon(contours)
print(hull)

temp_image = np.ones_like(binary_image_custom) * 255
temp_image = binary_image_custom.copy()
temp_image = image.copy()
# for pt in contours:
    # temp_image[pt[0], pt[1]] = 0

# hull.reverse()

# for point in hull:
    # cv2.circle(temp_image, point, 2, (0, 0), -1)

draw_mpp(temp_image, hull)

# # Display the image
cv2.imshow('Contour Points', temp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('poly_image.jpg', temp_image)
