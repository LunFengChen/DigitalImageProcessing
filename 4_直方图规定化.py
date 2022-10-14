import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_sk(img, ):
    width, height = img.shape
    img_array = img.ravel()

    pdf = 0
    sk = np.zeros(shape=(256, 1))

    for i in range(256):
        pdf += np.sum(img_array == i) / width / height
        sk[i] = np.round_(255 * pdf + 0.5)

    return sk


img = cv2.imread("./data/Cameraman.tif", 0)
img_regulation = cv2.imread("./data/lena.tif", 0)
sk1 = get_sk(img)
sk2 = get_sk(img_regulation)


plt.subplot(231), plt.imshow(img, "gray")
plt.subplot(234), plt.hist(img.ravel(), 256, [0, 256])

plt.subplot(232), plt.imshow(img_regulation, "gray")
plt.subplot(235), plt.hist(img_regulation.ravel(), 256, [0, 256])
plt.show()

def get_lut(sk1, sk2):
    sk1 = sk1.squeeze()
    sk2 = sk2.squeeze()
    diff = np.tile(sk1, (sk2.shape[0], 1))

    a = 0
    for i in np.nditer(sk2):
        diff[a] = np.abs(diff[a] - i)
        a += 1
    index = np.argmin(diff, axis=0)
    return index.astype(np.uint8)

print(get_lut(sk1, sk2))
