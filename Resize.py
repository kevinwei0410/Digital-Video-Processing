#0066C027
#Resize

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
img = Image.open('lenna.tif')
IMAGE = np.array(img)

Width_input, Height_input = 192, 108


def main():
    RE_IMAGE = Resize(IMAGE, (Width_input, Height_input))
    plt.imshow(RE_IMAGE, interpolation='nearest')
    plt.show()
def Resize(src, shape):
	height, width, channels = src.shape
	Width_input, Height_input = shape
	if ((Height_input == height) and (Width_input == width)):
		return src
	Image = np.zeros((Height_input, Width_input, channels), np.uint8)
	SX = float(width)/Width_input
	SY = float(height)/Height_input
	for k in range(channels):
		for J in range(Height_input):
			for I in range(Width_input):
				X = (I + 0.5) * SX - 0.5
				Y = (J + 0.5) * SY - 0.5
				X0 = int(np.floor(X))
				Y0 = int(np.floor(Y))
				X1 = min(X0 + 1, width - 1)
				Y1 = min(Y0 + 1, height - 1)
				value0 = (X1 - X) * src[Y0, X0, k] + (X - X0) * src[Y0, X0, k]
				value1 = (X1 - X) * src[Y1, X0, k] + (X - X0) * src[Y1, X0, k]
				Image[J, I, k] = int((Y1 - Y) * value0 + (Y - Y0) * value1)
	return Image

main()