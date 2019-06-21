# 改变图片对比度
import cv2
import numpy as np

def image_rename(old_name, output_path, operate_name, suq_number):
    """
    :param old_name: 待水平移动的图形名称（完全路径）
    :param output_path: 水平移动后保存的文件夹路径
    :param operate_name: 操作名称（用于图像命名）
    :param suq_number: 水平移动的序号
    :return: 返回新的名称（包含完全路径）
    """
    ima_name = old_name.split("\\").pop()
    ima_name = ima_name.split(".")
    str_suq_number = str(suq_number).rjust(3, '0')
    new_name = output_path + "\\" + ima_name[0] + "_" + operate_name + str_suq_number + "." + ima_name[1]
    return new_name


def image_edge(img):
    '''
    寻找图片最外层的轮廓，并将轮廓外层涂为黑色，轮廓内层涂为白色
    :param img: 待处理图像
    :return: 二值图
    '''
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 15, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black = cv2.cvtColor(np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.fillPoly(black, [hull], (255, 255, 255))
    return black

def image_contrast_2(ima_name, parameters, output_path, suq_number):
    ima = cv2.imdecode(np.fromfile(ima_name, dtype=np.uint8), -1)
    rows, cols, chunnel = ima.shape
    blank = np.zeros([rows, cols, chunnel], ima.dtype)  # np.zeros(img1.shape, dtype=uint8)
    ima_change = cv2.addWeighted(ima, parameters[0], blank, 1 - parameters[0], parameters[1])
    print(ima.shape)
    ima_edge = image_edge(ima)
    print(ima_edge.shape)
    for x in range(rows):
        for y in range(cols):
            if ima_edge[x,y][0] < 10:
                ima_change[x,y] = [0,0,0]
    cv2.imshow('test', ima)
    cv2.waitKey()
    cv2.imshow('test2', ima_change)
    cv2.waitKey()
    ima_new_name = image_rename(ima_name, output_path, "C", suq_number)
    cv2.imencode('.jpg', ima_change)[1].tofile(ima_new_name)


PATH = '.\\picture\\0.jpg'
image_contrast_2(PATH, [1.2,150], '.\\picture', 2)

'''
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 10, 255, cv2.THRESH_BINARY)
cv2.imshow('test', thresh)
cv2.waitKey()
# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 创建新的图像black
black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
cv2.imshow('test', black)
cv2.waitKey()


for cnt in contours:
    print(cnt.shape)
    # 轮廓周长也被称为弧长。可以使用函数 cv2.arcLength() 计算得到。这个函数的第二参数可以用来指定对象的形状是闭合的（True） ，还是打开的（一条曲线）
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    print(epsilon)
    # 函数approxPolyDP来对指定的点集进行逼近，cnt是图像轮廓，epsilon表示的是精度，越小精度越高，因为表示的意思是是原始曲线与近似曲线之间的最大距离。
    # 第三个函数参数若为true,则说明近似曲线是闭合的，它的首位都是相连，反之，若为false，则断开。
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print(approx.shape)
    # convexHull检查一个曲线的凸性缺陷并进行修正，参数cnt是图像轮廓。
    hull = cv2.convexHull(cnt)
    print(hull.shape)
    # 勾画图像原始的轮廓
    #cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    # 用多边形勾画轮廓区域
    #cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
    # 修正凸性缺陷的轮廓区域
    #cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)
    #cv2.polylines(black, [hull], True, (0, 0, 255))
    cv2.fillPoly(black, [hull], (255, 255, 255))

# 显示图像
cv2.imshow("hull", black)
print(black.shape)
cv2.waitKey()
cv2.destroyAllWindows()
'''

