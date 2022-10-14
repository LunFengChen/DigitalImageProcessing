import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("./data/Cameraman.tif", 0)

width, height = img.shape
# 生成灰度直方图
# 统计每个灰度出现次数
img_array = img.ravel()

pdf = 0
sk = np.zeros(shape=(256, 1))

for i in range(256):
    pdf += np.sum(img_array == i) / width / height
    sk[i] = np.round_(255 * pdf + 0.5)

img_new = img.copy()
for i in range(width):
    for j in range(height):
        img_new[i, j] = sk[img[i, j]]

# opencv 的标准函数
equ = cv2.equalizeHist(img)

plt.subplot(231), plt.imshow(img, "gray")
plt.subplot(234), plt.hist(img.ravel(), 256, [0, 256])

plt.subplot(232), plt.imshow(img_new, "gray")
plt.subplot(235), plt.hist(img_new.ravel(), 256, [0, 256])


plt.subplot(233), plt.imshow(equ, "gray")
plt.subplot(236), plt.hist(equ.ravel(), 256, [0, 256])

plt.show()
