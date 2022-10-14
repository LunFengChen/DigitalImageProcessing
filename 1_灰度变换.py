import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams.update({"font.size": 15})


def reverse(_img):
    return 255 - _img


def piecewise_linear(img, a=100, b=150, c=50, d=200, Mg=255, Mf=255):
    width, height = img.shape
    _img_new = img

    for i in range(width):
        for j in range(height):
            # print(img[i, j, k])
            if 0 <= img[i, j] < a:
                _img_new[i, j] = c / a * img[i, j]
            elif a <= img[i, j] < b:
                _img_new[i, j] = (d - c) / (d - a) * (img[i, j] - a) + c
            else:
                _img_new[i, j] = (Mg - d) / (Mf - b) * (img[i, j] - b) + d

    return np.round_(_img_new)


def log_trans(_img, c=255, a=10):
    return (c * np.log2(1 + _img) / np.log2(a)).astype('uint8')


def exp_trans(_img, c=255, gamma=5):
    return (c * _img ** gamma).astype('uint8')


if __name__ == '__main__':
    img = cv2.imread("./data/car.jpg", 0)
    # cv2.imshow("car_source", img)

    img_reverse = reverse(img)
    # cv2.imshow("car", img_new), cv2.waitKey()
    img_linear = piecewise_linear(img)

    img_log = log_trans(img)
    img_exp = exp_trans(img)

    plt.figure(figsize=(10, 10), dpi=200)
    # plt.subplot(221), plt.imshow(img), plt.title("原图")
    plt.subplot(221), plt.imshow(img_reverse, 'gray'), plt.title("反转变换")
    plt.subplot(222), plt.imshow(img_linear, 'gray'), plt.title("分段线性拉伸")
    plt.subplot(223), plt.imshow(img_log, 'gray'), plt.title("对数变换")
    plt.subplot(224), plt.imshow(img_exp, 'gray'), plt.title("指数")

    plt.show()
