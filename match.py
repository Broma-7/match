import numpy as np
import cv2,os
import time

def Get_npy(input_path):
    descriptors = []
    for (dirs, dirnames, filenames) in os.walk(input_path):
        for img_file in filenames:
            if img_file.endswith('npy'):
                descriptors.append(dirs+'/'+img_file)
    return descriptors


def SIFT_FLANN():
    sift = cv2.xfeatures2d.SURF_create(5000)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return sift,flann


def Detect_Matches(sift,flann,query,descriptors):
    query_kp, query_ds = sift.detectAndCompute(query, None)
    potential_culprits = {}
    for desc in descriptors:
        # 将图像query与特征数据文件的数据进行匹配
        matches = flann.knnMatch(query_ds, np.load(desc), k=2)

        # 清除错误匹配
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # 输出每张图片与目标图片的匹配特征数目
        print("img is %s ! matching rate is (%d)" % (desc, len(good)))
        potential_culprits[desc] = len(good)

    # 获取最多匹配数目的图片
    max_matches = None
    potential_suspect = None
    for culprit, matches in potential_culprits.items():
        if max_matches == None or matches > max_matches:
            max_matches = matches
            potential_suspect = culprit
    print("potential suspect is %s" % potential_suspect.replace("npy", "").upper())


if __name__ == '__main__':
    start = time.time()
    query = cv2.imread('./2_180angle.png', 0)
    input_path = 'D:\python\pythonProject2\image'
    descriptors = Get_npy(input_path)                # 获取特征数据文件
    sift, flann = SIFT_FLANN()                       # 使用SIFT算法检查图像的关键点和描述符，创建FLANN匹配器
    Detect_Matches(sift, flann, query, descriptors)  # 检测并匹配
    end = time.time()
    print("特征点匹配运行时间:%.2f秒" % (end - start))
