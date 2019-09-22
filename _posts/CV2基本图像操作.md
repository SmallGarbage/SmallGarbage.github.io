---
layout:     post   				    # 使用的布局（不需要改）
title:      简单图形处理cv2				# 标题 
subtitle:   opencv, 数据  提取 #副标题
date:       2019-09-22 				# 时间
author:     程全海 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 图片
---
#### 数据提取-图像

- ``cv2.IMREAD_COLOR``:彩色图像
- `cv2.IMREAD_GRAYSTYLE`:灰色图像

```python
import cv2  # opencv读取的格式是BGR
img = cv2.imead("cat.jpg")
img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSTYLE)
cv2.imshow("reslut", img) # 显示读取的图像
cv2.waitKey(0) # 图片持续时间

# print(img.shape) ---> (414,500,3)  # 三通道

img2 = cv2.imwrite("mycat.png", img) # 将图片保存以特定的格式
```

#### 数据读取-视频

- `cv2.VideoCapture`:可以捕获摄像头，用数字控制不同设备，例如 0,1
- 如果是视频文件，直接指定路径即可

```python
vc = cv2.VideoCapture("test.mp4")
# 检查是否打开正确
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
```

#### 截取部分图像数据

```python
img = cv2.imread("cat.jpg")
cat = img[0:50,0:200]
cv2.imshow("result", cat)
cv2.waitKey(0)
```

#### 颜色通道提取

```python
b, g, r = cv2.split(img) # 颜色通道提取
img = cv2.merge((b,g,r)) # 颜色通道合成
```

#### 边界填充

- BORDER_REPLICATE:复制法，也就是复制最边缘像素
- BORDER_REFLECT:反射法，对感兴趣图形中的像素在两边进行复制：fedcb|bcdef|fedcb
- BORDER_REFLECT_101:反射法，以最边缘像素为轴，对称复制
- BORDER_WRAP:外包装法
- BORDER_CONSTANT：常量法，常数值填充

```python
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50) # 设置边界填充框大小
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, righ_size, boderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)

# 将生成图像画出来
import matplotlib.pyplot as plt

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()
```

![](<https://github.com/SmallGarbage/SmallGarbage.github.io/blob/master/394f62ba6ef65bded33a338b86fe834.png>)

#### 图像融合

```python
import cv2
img_cat = cv2.imread("cat.jpg")
img_dog = cv2.imread("dog.jpg")

img_dog = cv2.resize(img_dog, (500, 414)) # 将图片统一成一致大小
img_cat = cv2.resize(img_dog, (500, 414))

res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0) # 设置权重相融合

res = cv2.resize(img, (0, 0), fx=4, fy=4) # 设置横纵坐标比例
plt.imshow(res)
```

#### 灰度图

```python
import cv2
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

#### 图像阈值

**ret,dst = cv2.threshold(src, thresh, maxval, type)**

- src: 输入图，只能输入单通道图像，通常来说灰度图
- dst: 输出图
- thresh: 阈值
- maxval: 当像素值超过某个阈值，或小于某个值，人为赋予的词
- type: 二值化操作的类型，包括如下五种：
- cv2.THRESH_BINARY: 超过阈值取最大值，否则取0
- cv2.THRESH_BINARY_INV: 以上翻转
- cv2.THRESH_TOZERO: 大于阈值部分不改变，否则设为0
- cv2.THRESH_TOZERO_INV: 以上翻转
- cv2.THRESH_TRUNC: 大于阈值部分设为阈值，否则不变

```python
ret, thresh1 = cv2.threshole(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(len(images)):
    plt.subplot(2, 3, i+1),plt.imshow(images[i],"gray")  # 创建小子图，每个子图显示
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()    
```

#### 图形平滑

```python
img = cv2.imread("lenaNoise.png")

cv2.imshow("img", img)
cv2.waitKey(0)
# 均值滤波
blur = cv2.blur(img, (3,3)) # 3x3 卷积核，求卷积核的平均值作为新值
cv2.imshow("blur", blur)
cv2.wiatKey(0)

# 方框滤波
# 等同于 均值滤波，可以归一化
box = cv2.boxFilter(img,-1,(3,3),normalize=False)
cv2.imshow("box",box)
cv2.waitKey(0)

# 高斯滤波
# 高斯滤波的卷积核里的数值是满足高斯分布的，相当于重视中间的
aussian = cv2.Gaussian(img,(5,5),1)
cv2.imshow("aussian",aussian)
cv2.waitKey(0)

# 中值滤波
# 先排序，然后用中值替代
median = cv2.medianBlur(img, 5)  # 中值滤波
cv2.imshow('median', median)
cv2.waitKey(0)

# 展示所有的
res = np.hstack((blur,aussian,median))
cv2.imshow('median vs average', res)
cv2.waitKey(0)
```

#### 形态学-腐蚀操作

```python
img = cv2.imread("dige.png")
cv2.imshow("img",img)
cv2.wiatKey(0)
# 定义一个卷积核
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow("erosion", ersion)
cv2.wiatKey(0)
```

#### 形态学-膨胀操作

```python
img = cv2.imread("dige.png")
cv2.imshow("img",img)
cv2.wiatKey(0)
# 定义一个卷积核
kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("dilate", dilate)
cv2.wiatKey(0)
```

#### 开运算与闭运算

```python
# 开：先腐蚀，再膨胀
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8) 
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening', opening)
cv2.waitKey(0)

# 闭：先膨胀，再腐蚀
img = cv2.imread('dige.png')

kernel = np.ones((5,5),np.uint8) 
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing', closing)
cv2.waitKey(0)
```

#### 梯度运算

```pyton
# 梯度=膨胀-腐蚀
pie = cv2.imread('pie.png')
kernel = np.ones((7,7),np.uint8) 
dilate = cv2.dilate(pie,kernel,iterations = 5)
erosion = cv2.erode(pie,kernel,iterations = 5)

res = np.hstack((dilate,erosion))

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('gradient', gradient)
cv2.waitKey(0)
```

#### 礼帽与黑帽

- 礼帽 = 原始输入-开运算结果
- 黑帽 = 闭运算-原始输入

```pyton
#礼帽
img = cv2.imread('dige.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)

#黑帽
img = cv2.imread('dige.png')
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat ', blackhat )
cv2.waitKey(0)
```

#### 图像梯度-Soble算子

```pyton
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely) 

sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow("result",sobelxy)
```

#### 图像轮廓

#### cv2.findContours(img,mode,method)

- RETR_EXTERNAL ：只检索最外面的轮廓；
- RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
- RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
- RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;

method:轮廓逼近方法

- CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
- CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分

为了更高的准确率，使用二值图像

```python
img = cv2.imread("coutours.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary, countours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# 绘制轮廓
drow_img = img.copy()
res = cv2.drawContours(drow_img, contours, -1 (0,0,255),2)

# 边界矩形
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w, y+h),(0, 255,0),2)
# img就是在img原图上绘制完外接矩形的图形啊
```

#### 模板匹配

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("messi.png",0)
img2 = img.copy()
template = cv2.imread("face.png",0)
w, h = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    res = cv2.matchTemplate(img,template,method)
    mn_val, max_val,min_loc,max_loc = cv2.minMaxloc(res)
    if method in  [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)  
    cv2.rectangle(img, top_left, bottom_right,255,2)
    
    plt.subplot(121),plt.imshow(res, cmap="gray")
     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('method: ' + meth)
	plt.show()
```

