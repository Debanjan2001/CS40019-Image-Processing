import cv2

class Part1Processor:

    def __init__(self) -> None:
        pass

    def read_image(self, img_path):
        self.image_path = img_path
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        return self.gray_image
    

def main():
    image_path = 'restore.png'
    part1Solver = Part1Processor()
    gray_image = part1Solver.read_image(image_path)

if __name__ == '__main__':
    main()