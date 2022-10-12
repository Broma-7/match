import numpy as np
import cv2,os
import time
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import sys

def Get_npy(input_path):
    descriptors = []
    for (dirs, dirnames, filenames) in os.walk(input_path):
        for img_file in filenames:
            if img_file.endswith('npy'):
                descriptors.append(dirs+'/'+img_file)
    return descriptors


def SIFT_FLANN():
    sift = cv2.xfeatures2d.SURF_create(3000)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=100)
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
            if m.distance < 0.9 * n.distance:
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
    print("potential suspect is %s" % potential_suspect.replace(".npy", ""))
    potential_suspect_jpg = potential_suspect.replace(".npy", ".jpg")
    str = tk.StringVar()  # StringVar是一个很强大的类，可以辅助控件动态改变值
    label_result = tk.Label(window, textvariable=str, fg='red', font=('宋体', 20))  # 用于显示警告信息
    label_result.place(x=90, y=550, anchor='nw')
    str.set(f"potential suspect is {potential_suspect_jpg}")
    global photo2
    global photo2_resized
    global tk_image2
    global img2
    photo2 = Image.open(potential_suspect_jpg)  # 之前留的备份
    # 获取图像的原始大小
    w2, h2 = photo2.size
    # 缩放图像让它保持比例，同时限制在一个矩形框范围内
    photo2_resized = resize(w2, h2, w_box, h_box, photo2)
    tk_image2 = ImageTk.PhotoImage(photo2_resized)  # 放在window上
    img2 = tk.Label(window, image=tk_image2)  # 放在window上
    img2.place(x=800-170, y=100+50)  # 固定
def match(path):
    global count
    query = cv2.imread(path)
    descriptors = Get_npy(input_path)                # 获取特征数据文件
    end1 = time.time()
    if count == 1:
        str = tk.StringVar()  # StringVar是一个很强大的类，可以辅助控件动态改变值
        label_result = tk.Label(window, textvariable=str, fg='red', font=('宋体', 20))  # 用于显示警告信息
        label_result.place(x=90, y=470, anchor='nw')
        str.set("获取特征数据文件运行时间:%.2f秒" % (end1 - start))
        # print("获取特征数据文件运行时间:%.2f秒" % (end1 - start))
        count = 0
    sift, flann = SIFT_FLANN()                       # 使用SIFT算法检查图像的关键点和描述符，创建FLANN匹配器
    Detect_Matches(sift, flann, query, descriptors)  # 检测并匹配
    end2 = time.time()
    str = tk.StringVar()  # StringVar是一个很强大的类，可以辅助控件动态改变值
    label_result = tk.Label(window, textvariable=str, fg='red', font=('宋体', 20))  # 用于显示警告信息
    label_result.place(x=90, y=500, anchor='nw')
    str.set("检测并匹配运行时间:%.2f秒" % (end2 - end1))
    print("检测并匹配运行时间:%.2f秒" % (end2 - end1))
    print("总时间特征点匹配运行时间:%.2f秒" % (end2 - start))
def resize(w, h, w_box, h_box, pil_image):
    # 对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)
    # w,h是原图像的大小， w_box， h_box为指定的大小，返回修改好的图片


def Readfilepath():
    global fileindex, fatherpath, files, file_num
    filepath = filedialog.askopenfilename()  # 获取图片全部路径
    fatherpath = os.path.dirname(filepath)  # 获取该路径的上一级路径
    filename = os.path.basename(filepath)  # 获取该路径下的文件名
    files = os.listdir(fatherpath)  # 该路径下的所有文件并生成列表
    file_num = len(files)
    fileindex = files.index(filename)  # 获取当前文件的索引值
    checkout(filepath)  # 调用之前嵌套定义的original


def checkout(path):
    # 输入图像
    # img = cv.imread(path)  # 读入图像
    # img_copy = img.copy()
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转灰度图

    def original():  # 显示原始图片
        # 以一个PIL图像对象打开
        global photo1
        global photo1_resized
        global tk_image1
        global img1
        photo1 = Image.open(path)
        # 获取图像的原始大小
        w1, h1 = photo1.size
        # 缩放图像让它保持比例，同时限制在一个矩形框范围内
        photo1_resized = resize(w1, h1, w_box, h_box, photo1)
        tk_image1 = ImageTk.PhotoImage(photo1_resized)  # 放在window上
        img1 = tk.Label(window, image=tk_image1)  # 放在window上
        img1.place(x=80-10, y=100+50)  # 固定

    def after():  # 显示批改后的图片

        # 以一个PIL图像对象打开
        global photo2
        global photo2_resized
        global tk_image2
        global img2
        photo2 = Image.open("./8.jpg")  # 之前留的备份
        # 获取图像的原始大小
        w2, h2 = photo2.size
        # 缩放图像让它保持比例，同时限制在一个矩形框范围内
        photo2_resized = resize(w2, h2, w_box, h_box, photo2)
        tk_image2 = ImageTk.PhotoImage(photo2_resized)  # 放在window上
        img2 = tk.Label(window, image=tk_image2)  # 放在window上
        img2.place(x=800, y=100)  # 固定

    original()
    return after


def check():
    global fileindex, fatherpath, files
    filepath1 = os.path.join(fatherpath, files[fileindex])  # 获取全路径
    match(filepath1)

def back():
    global fileindex, fatherpath, files, file_num
    fileindex -= 1  # 获取上一个图片下标
    if fileindex == -1:
        fileindex = file_num - 1  # 如果已经为第一个则获取最后一个的下标
    filepath2 = os.path.join(fatherpath, files[fileindex])  # 获取全路径
    checkout(filepath2)  # 调用original(),transform()显示图像


def next():
    global fileindex, fatherpath, files, file_num
    fileindex += 1  # 获取下一张图片的坐标值
    if fileindex == file_num:
        fileindex = 0  # 如果为最后一张则取第一张的
    filepath3 = os.path.join(fatherpath, files[fileindex])  # 获取全路径
    checkout(filepath3)  # 调用original(),transform()显示图像


def exit_():
    sys.exit(0)  # 退出


if __name__ == '__main__':
    start = time.time()
    input_path = './Medecine'
    count = 1
    wh = 15
    w_box = 400
    h_box = 640
    window = tk.Tk()  # 定义窗口
    window.title("window")  # 命名窗口
    window.geometry("1080x720")  # 定义大小
    title0 = tk.Label(window, text="图片特征匹配", font=("华文行楷", 45), fg="blue")  # 标题标签
    title0.place(x=450-100-30+15, y=20)  # 固定
    # 批改前
    label1 = tk.Label(window, text="original", font=("Arial Bold", 30))
    label1.place(x=200-50+20+20, y=50+50)
    # 批改后
    label2 = tk.Label(window, text="match", font=("Arial Bold", 30))
    label2.place(x=950-170, y=50+50)
    button1 = tk.Button(window, text="读取文件夹路径", height=2, width=15, command=Readfilepath)  # 注意不要加(),加括号会直接触发函数
    button1.place(x=80 + wh, y=650-40)  # wh是我调位置用的一个值 此时wh=15
    button2 = tk.Button(window, text="开始匹配", height=2, width=15, command=check)
    button2.place(x=280 + wh, y=650-40)
    button3 = tk.Button(window, text="上一张", height=2, width=15, command=back)
    button3.place(x=280 + 200 + wh, y=650-40)
    button4 = tk.Button(window, text="下一张", height=2, width=15, command=next)
    button4.place(x=280 + 200 + 200 + wh, y=650-40)
    button5 = tk.Button(window, text="退出", height=2, width=15, command=exit_)
    button5.place(x=280 + 200 + 200 + 200  + wh, y=650-40)
    photo = Image.open("H://njupt_ps.png")
    w, h = photo.size
    photo_resize = resize(w, h, 70, 70, photo)
    photo_ = ImageTk.PhotoImage(photo_resize)
    theLabel = tk.Label(window,
                        justify=tk.LEFT,  # 对齐方式
                        image=photo_,  # 加入图片
                        compound=tk.CENTER,  # 关键:设置为背景图片
                        )
    theLabel.place(x=720-30+15, y=20)
    window.mainloop()



