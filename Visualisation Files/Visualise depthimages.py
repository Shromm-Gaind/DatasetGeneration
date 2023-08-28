import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def main():

    filename = ''  # <-- hardcoding the file path here

    print("Converting " + filename)

    srcImg = cv2.imread(filename, cv2.IMREAD_COLOR)
    srcHeight, srcWidth, srcChannels = srcImg.shape
    img32f = np.zeros((srcHeight, srcWidth, 1), dtype = "float32")

    for y in range(0, srcHeight):
        for x in range(0, srcWidth):
            pixelDepthMM = srcImg[y, x][2] + srcImg[y, x][1] * 256 + srcImg[y, x][0] * 256 * 256
            pixelDepthM = pixelDepthMM / 1000
            img32f[y, x] = pixelDepthM

    minVal = np.amin(img32f)
    maxVal = np.amax(img32f)
    draw = cv2.convertScaleAbs(img32f, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))

    # using Matplotlib to display the image
    plt.gray()
    plt.imshow(draw)
    plt.show()

    fileNameAndExt = os.path.splitext(filename)
    cv2.imwrite(fileNameAndExt[0]  + '_converted' + fileNameAndExt[1], draw)

if __name__ == "__main__":
    main()
