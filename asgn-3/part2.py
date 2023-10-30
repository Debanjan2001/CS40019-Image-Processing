import cv2
import numpy as np
import matplotlib.pyplot as plt

class Part2Processor:

    def __init__(self) -> None:
        self.dx = [-1, +1, 0, 0]
        self.dy = [0, 0, -1, +1]
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
        ]

    def read_image(self, img_path):
        self.image_path = img_path
        self.rgb_image = cv2.imread(self.image_path)
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self.gray_image
    
    def apply_otsu_thresholding(self, image):
        print("[Started] Otsu Thresholding...")
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

            if value >= final_value:
                final_thresh = t
                final_value = value

        final_img = image.copy()
        # print(final_thresh)
        final_img[image > final_thresh/3] = 255
        final_img[image <= final_thresh/3] = 0
        print("[Finished] Otsu Thresholding.")
        return final_img
    
    def perform_bfs(self, img, sx, sy, vis):
        vis[(sx,sy)]=True;
        queue = [(sx, sy)]
        component = []
        h, w = img.shape[0], img.shape[1]
        def inside(x, y):
            return x<h and x>=0 and y<w and y>=0
        
        while len(queue) > 0:
            x,y = queue.pop(0)
            for i in range(len(self.dx)):
                nx = x+self.dx[i]
                ny = y+self.dy[i]
                component.append((x,y))
                if (nx,ny) in vis:
                    continue
                if not inside(nx, ny):
                    continue
                if img[nx][ny] != img[x][y]:
                    continue 
                vis[(nx,ny)] = True
                queue.append((nx,ny))
        return component
    
    def apply_connected_component_labelling(self, binary_image):
        print("[Started] Connected Component Labelling...")
        image = binary_image.copy()
        image = image // 255
        h, w = image.shape
        # print(h,w)
        components = []
        vis = dict()
        for i in range(h):
            for j in range(w):
                if image[i][j] == 0:
                    continue
                if (i, j) in vis:
                    continue
                vis[(i,j)]=True
                comp = self.perform_bfs(image, i, j, vis)
                components.append(comp)

        # print(len(components))
        print("[Finished] Connected Component Labelling.")
        return components
    
    def apply_color_and_save(self, components):
        for i, c in enumerate(components):
            col = self.colors[i % len(self.colors)]
            for (x,y) in c:
                self.rgb_image[x][y] = col

        plt.imsave('part2_labelled_image.jpg', self.rgb_image)

    
def main():
    image_path = 'connect.jpg'
    part2Solver = Part2Processor()
    gray_image = part2Solver.read_image(image_path)

    otsu_image = part2Solver.apply_otsu_thresholding(gray_image)
    plt.imsave('part2_otsu_image.jpg', otsu_image, cmap='gray')

    components = part2Solver.apply_connected_component_labelling(otsu_image)
    part2Solver.apply_color_and_save(components)


if __name__ == '__main__':
    main()