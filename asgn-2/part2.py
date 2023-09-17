"""
Name - Debanjan Saha
Roll - 19CS30014
Assignment - 2
Part-2: MPP algorithm
Comments: Could not figure out the exact bug due to which the bounded region is not fitted accurately
"""
import cv2
import numpy as np

class LibMPP:
    def __init__(self):
        pass
    
    def read_image(self, img_path):
        self.image_path = img_path
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self.gray_image

    def apply_adaptive_threshold(self, image, block_size, constant):
        print("Please wait! Applying Adaptive Threshold...")
        height, width = image.shape
        binary_image = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = image[i:i+block_size, j:j+block_size]
                mean_intensity = np.mean(block)
                threshold = mean_intensity - constant
                binary_block = (block > threshold).astype(np.uint8) * 255
                binary_image[i:i+block_size, j:j+block_size] = binary_block
        print("Done!")
        return binary_image
    
    def find_points(self, binary_image):
        rows,cols = binary_image.shape
        points = []
        for i in range(rows):
            for j in range(cols):
                # 0 for foreground and 255 for background
                if binary_image[i][j] == 0:
                    points.append((i, j))
        return points

    def sort_points_anticlockwise(self, points):
        def find_slope(p1, p2):
            if p1[0] == p2[0]:
                if(p2[1] > p1[1]):
                    return float('inf')
                else:
                    return float('-inf')
            else:
                return 1.0*(p2[1]-p1[1])/(p2[0]-p1[0])

        points.sort(key=lambda x:[x[0],x[1]])
        start = points.pop(0)
       
        points.sort(key=lambda p: (p[0],p[1]))
        lower_half, upper_half = [], []
        for pt in points:
            slope = find_slope(start, pt)
            if slope >= 0:
                upper_half.append(pt)
            else:
                lower_half.append(pt)
        
        temp_points = [start]
        temp_points.extend(lower_half)
        upper_half.sort(key=lambda p: (p[0],-p[1]), reverse=True)
        temp_points.extend(upper_half)
        points = temp_points
        return points

    def get_sign_value(self, p1, p2, p3):
        return (p3[0]-p2[0])*(p1[1]-p2[1]) - (p3[1]-p2[1])*(p1[0]-p2[0])
    
    def getConvexArray(self, points):
        is_convex = [True for i in range(len(points))]
        for i in range(len(points)):
            if self.get_sign_value(points[i-1], points[i], points[(i+1)%len(points)]) < 0:
                is_convex[i] = False
            else:
                is_convex[i] = True
        return is_convex
    
    def apply_minimum_perimeter_polygon(self, points):
        print("Please wait! Calculating MPP...")
        points = self.sort_points_anticlockwise(points)
        is_convex = self.getConvexArray(points)

        # Apply the actual algorithm as mentioned in slides
        n = len(points)
        wc, bc, vl, vk = 0, 0, 0, 0
        mpp_indices = set()
        while(vk<n):
            sgn1 = self.get_sign_value(points[vl], points[wc], points[vk])
            sgn2 = self.get_sign_value(points[vl], points[bc], points[vk])
            if(sgn1 > 0):
                vl = wc
                wc = vl
                bc = vl
                vk = vl+1
            elif(sgn1 <= 0 and sgn2>=0):
                if(is_convex[vk]):
                    wc = vk
                else:
                    bc = vk
                vk = vk+1
            elif(sgn2<0):
                vl = bc
                wc=vl
                bc = vl
                vk = vl+1
            # print(vl)
            mpp_indices.add(vl)

        mpp = [points[i] for i in sorted(list(mpp_indices))]
        print("Done!")
        return mpp

    def draw_on_image(self, image, mpp):
        mpp = np.array(mpp)
        cv2.polylines(image, [mpp], isClosed=True, color=(0, 255, 0), thickness=2)
        return image
    
    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")
        return

def main():
    image_path = 'poly.jpg'
    mpp_finder = LibMPP()
    gray_image = mpp_finder.read_image(image_path)
    binary_thresholded_image = mpp_finder.apply_adaptive_threshold(gray_image, 4, 127.4) # Obtained after trial and error
    mpp = mpp_finder.apply_minimum_perimeter_polygon(mpp_finder.find_points(binary_thresholded_image))
    mpp_finder.draw_on_image(gray_image, mpp)
    mpp_finder.save_image(gray_image, 'mpp.jpg')

if __name__ == '__main__':
    main()