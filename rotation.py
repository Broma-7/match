import cv2
import numpy as np
from matplotlib import pyplot as plt

def FLANN_MIN_MATCH_COUNT():
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('image/1.png', 0)
    img2 = cv2.imread('./2_180angle.png',0)

    sift = cv2.xfeatures2d.SURF_create(2500)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # 单应性
    if len(good)>MIN_MATCH_COUNT:
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()           # ravel()展平，并转成列表

        h,w = img1.shape
        # pts是图像img1的四个顶点
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)        # 计算变换后的四个顶点坐标位置

        # 根据四个顶点坐标位置在img2图像画出变换后的边框
        img2 = cv2.polylines(img2,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = None,
                       matchesMask = matchesMask[:100],
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:100],None,**draw_params)
    return img3
if __name__ == '__main__':
    img3 = FLANN_MIN_MATCH_COUNT()
    # plt.imshow(img3, 'gray')
    # plt.show()
    cv2.imshow('gray',img3)
    cv2.waitKey(0)
