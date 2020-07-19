#0066C027
#Move

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = Image.open('lenna.tif')
image = np.array(img)

move_x = 200
move_y = 1000
image2 = np.zeros([image.shape[0]+move_x, image.shape[1]+move_y, image.shape[2]], np.uint8)
def main():
    new_img = move()
    plt.imshow(image, interpolation='nearest')
    plt.imshow(new_img, interpolation='nearest')
    plt.show()

def move():
    kernel = np.array([[1,0,move_x],[0,1,move_y],[0,0,1]])
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            new_h, new_w, _= kernel.dot(np.array([h, w, 1]))
            if(0<=new_h<image2.shape[0] and 0 <= new_w < image2.shape[1]):
                image2[new_h, new_w] = image[h, w]
    return image2
main()