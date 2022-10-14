import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



def globalEqualHist(image):
    dst = cv.equalizeHist(image)
    return dst

def localEqualHist(image):
    clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7,7))
    dst = clahe.apply(image)
    return dst


img = cv.imread("./data/handsomeboy.png", 0)
img1 = globalEqualHist(img)
img2 = localEqualHist(img)

plt.subplot(131), plt.imshow(img, "gray")
plt.subplot(132), plt.imshow(img1, "gray")
plt.subplot(133), plt.imshow(img2, "gray")

plt.show()
