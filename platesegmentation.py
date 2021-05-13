from cv2 import cv2
import numpy as np
import os

lower_blue = np.array([ 90, 90, 0 ])
upper_blue = np.array([ 120, 210, 250 ])

img = cv2.imread('thre_res.png')

n = 0.8
sp = img.shape
height = round(n * sp[ 0 ])
weight = round(n * sp[ 1 ])
new_img = cv2.resize(img, (weight, height))
cv2.imshow('new_img', new_img)

hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
mark = cv2.inRange(hsv, lower_blue, upper_blue)
mark = cv2.bitwise_not(mark)
cv2.imshow("mark", mark)

# 腐蚀和膨胀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 5))  # 定义矩形结构元素

img_erode = cv2.erode(mark, kernel, iterations=1)
img_dilated = cv2.dilate(mark, kernel, iterations=3)
cv2.imshow('erode', img_dilated)

contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[ i ])
    print(area)
    if area > 4000:

        rect = cv2.minAreaRect(contours[ i ])  # 提取矩形坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = abs(abs(rect[ 2 ]) - 45)
        length = max(rect[ 1 ])
        wideth = min(rect[ 1 ])
        bili = length / (wideth + 0.01)

        area = rect[ 1 ][ 0 ] * rect[ 1 ][ 1 ]

        print(area)
        print(bili)
        print(rect[ 2 ])
        if area > 20000 or angle < 30:
            continue

        if bili > 3.3 or bili < 2.5:
            continue
        print(area)
        print(bili)
        print(rect[ 2 ])

        cv2.drawContours(new_img, [ box ], 0, (0, 0, 255), 2)
        if abs(rect[ 2 ]) < 45:
            a = rect[ 0 ][ 0 ] - 0.5 * rect[ 1 ][ 0 ]
            b = rect[ 0 ][ 0 ] + 0.5 * rect[ 1 ][ 0 ]
            c = rect[ 0 ][ 1 ] - 0.5 * rect[ 1 ][ 1 ]
            d = rect[ 0 ][ 1 ] + 0.5 * rect[ 1 ][ 1 ]

        else:
            a = rect[ 0 ][ 0 ] - 0.5 * rect[ 1 ][ 1 ]
            b = rect[ 0 ][ 0 ] + 0.5 * rect[ 1 ][ 1 ]
            c = rect[ 0 ][ 1 ] - 0.5 * rect[ 1 ][ 0 ]
            d = rect[ 0 ][ 1 ] + 0.5 * rect[ 1 ][ 0 ]

        # print(a,b,c,d)
        license_image = new_img[ round(c):round(d), round(a):round(b) ]
n_license = 3
sp = license_image.shape
print(sp)
weight_license = round(n_license * license_image.shape[ 1 ])
height_license = round(n_license * license_image.shape[ 0 ])

license_image = cv2.resize(license_image, (weight_license, height_license))

license_image_gray = cv2.cvtColor(license_image, cv2.COLOR_BGR2GRAY)

ret, license_image_erzhi = cv2.threshold(license_image_gray, 160, 255, cv2.THRESH_BINARY)

cv2.imshow('img', license_image_erzhi)

img_erode = cv2.erode(license_image_erzhi, kernel, iterations=1)
img_dilated = cv2.dilate(img_erode, kernel, iterations=2)
cv2.imshow('erode', img_dilated)

path = 'C:/Users/38601/Desktop/platenumber/'

if not os.path.exists(path):
    os.mkdir(path)

location_list = [ ]
contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = 0

for i in range(len(contours)):

    x, y, w, h = cv2.boundingRect(contours[ i ])
    # 在img图像画出矩形，(x, y), (x + w, y + h)是矩形坐标，(0, 255, 0)设置通道颜色，2是设置线条粗度
    area = w * h
    length = max(w, h)
    wideth = min(w, h)
    bili = length / (wideth + 0.01)
    if area < 1000 or area > 10000 or bili > 3:
        continue

    location_list.append((x, y, w, h))
    count = count + 1

print(count)


def takefirst(elem):
    return elem[ 0 ]


# 指定第一个元素排序
location_list.sort(key=takefirst)
print(location_list)

for i in range(0, count):
    license_image_1 = img_dilated[ location_list[ i ][ 1 ]:location_list[ i ][ 1 ] + location_list[ i ][ 3 ],
                      location_list[ i ][ 0 ]: location_list[ i ][ 0 ] + location_list[ i ][ 2 ] ]
    file_name = 'img' + str(i) + '.jpg'
    cv2.imwrite(path + file_name, license_image_1)
    cv2.rectangle(license_image_erzhi, (location_list[ i ][ 0 ], location_list[ i ][ 1 ] - 10), (
        location_list[ i ][ 0 ] + location_list[ i ][ 2 ], location_list[ i ][ 1 ] + location_list[ i ][ 3 ]),
                  (255, 255, 255), 4)

cv2.imshow('result', license_image_erzhi)
cv2.waitKey(0)
