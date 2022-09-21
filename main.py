import cv2

import time
def sift(img):
    start = time.time()
    sift = cv2.xfeatures2d.SIFT_create(6500)
    keypoints = sift.detect(img, None)

    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0, 0, 255))
    end = time.time()
    cv2.imshow('Detected SIFT keypoints', img2)
    cv2.waitKey(0)

    return end - start

# it is a test
def surf(img):
    start = time.time()
    surf = cv2.xfeatures2d.SURF_create(6500)
    # 设置是否要检测方向
    # surf.setUpright(True)
    #
    # # 输出设置值
    # print(surf.getUpright())
    # 找到关键点和描述符
    key_query, desc_query = surf.detectAndCompute(img, None)
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, key_query, img2, (0, 0, 255),4)
    end = time.time()
    cv2.imshow('Detected SURF keypoints', img2)
    cv2.waitKey(0)
    print(surf.descriptorSize())
    return end - start


image = cv2.imread("image/pill.jpeg")
# print('sift_time:', sift(image))
print('surf_time:', surf(image))
# print('OpenCv Version:',cv2.__version__)
