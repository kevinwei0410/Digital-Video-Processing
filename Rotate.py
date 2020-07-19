#0066C027
#Rotate

import math   
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = Image.open('lenna.tif')
image = np.array(img)


degree = 30

def main():
    iLRotate30 = LRotate(image,degree)
    plt.imshow(image, interpolation='nearest')
    print("Rotate Degree:", degree)
    plt.imshow(iLRotate30, interpolation='nearest')
    plt.show()
    
def LRotate(image,angle):
    h, w, c = image.shape
    anglePi = angle*math.pi/180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    X1 = math.ceil(abs(0.5*h*cosA + 0.5*w*sinA))
    X2 = math.ceil(abs(0.5*h*cosA - 0.5*w*sinA))
    Y1 = math.ceil(abs(-0.5*h*sinA + 0.5*w*cosA))
    Y2 = math.ceil(abs(-0.5*h*sinA - 0.5*w*cosA))
    H = int(2*max(Y1,Y2))
    W = int(2*max(X1,X2))
    size = (W+1,H+1,3)
    iLRotate = np.zeros(size, np.uint8)
    for i in range(h):
        for j in range(w):            
            x = int(cosA*i-sinA*j-0.5*w*cosA+0.5*h*sinA+0.5*W)
            y = int(sinA*i+cosA*j-0.5*w*sinA-0.5*h*cosA+0.5*H)
            iLRotate[x,y] = image[i,j]
    iLRotate = meanFilter(iLRotate)
    return iLRotate 
def meanFilter(im):
    img = im
    w = 2
    for i in range(2,im.shape[0]-2):
        for j in range(2,im.shape[1]-2):
            for c in range(im.shape[2]):
                block = im[i-w:i+w+1, j-w:j+w+1, c]
                m = np.mean(block,dtype=np.float32)
                img[i][j][c] = int(m)
    return img

main()