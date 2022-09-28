import cv2
import numpy as np
import os
import time

def get_img(input_path):
    image_paths = []
    for (dirs, dirnames, filenames) in os.walk(input_path):
        for img_file in filenames:
            ext = ['.jpg','.png','.jpeg','.tif']
            if img_file.endswith(tuple(ext)):
                image_paths.append(dirs+'/'+img_file)
    return image_paths


def save_descriptor(sift,image_path):
    if image_path.endswith("npy"):
        return
    img = cv2.imread(image_path, 0)
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # 设置文件名并将特征数据保存到npy文件
    descriptor_file = image_path.replace(image_path.split('.')[-1], "npy")
    np.save(descriptor_file, descriptors)


if __name__=='__main__':
    start = time.time()
    input_path = './image'
    image_paths = get_img(input_path)
    sift = cv2.xfeatures2d.SURF_create(2500)
    for image_path in image_paths:
        save_descriptor(sift, image_path)
    print('done!')
    end = time.time()
    print("保存特征模型运行时间:%.2f秒" % (end - start))
