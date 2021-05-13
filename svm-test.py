import cv2
import numpy as np
import sys
import os

import json
import svm

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


class CardPredictor:
    def __init__(self):
        # 车牌识别的部分参数保存在json
        f = open('config.json')
        j = json.load(f)
        for c in j[ "config" ]:
            if c[ "open" ]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')

    def load_svm(self):
        # 识别英文字母和数字
        self.model = svm.SVM(C=1, gamma=0.5)  # SVM(C=1, gamma=0.5)
        # 识别中文
        self.model.load("module\\svm.dat")

    def predict(self, path, resize_rate=1):
        predict_result = [ ]
        part_cards = [ ]
        roi = None
        card_color = None
        for filename in os.listdir(path):
            print(filename)
            digit_img = imreadex(filename)
            print(digit_img)
            part_cards.append(digit_img)
        for i, part_card in enumerate(part_cards):
            part_card_old = part_card
            w = part_card.shape[ 1 ] // 3
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[ 0, 0, 0 ])
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
            cv2.imshow("part_card", part_card)
            cv2.waitKey(0)
            part_card = svm.preprocess_hog([ part_card ])
            resp = self.model.predict(part_card)  # 调用svm模型
            charactor = chr(resp[ 0 ])
            # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
            if charactor == "1" and i == len(part_cards) - 1:
                if part_card_old.shape[ 0 ] / part_card_old.shape[ 1 ] >= 8:  # 1太细，认为是边缘
                    print(part_card_old.shape)
                    continue
            predict_result.append(charactor)

        return predict_result  # 识别到的字符、定位的车牌图像


if __name__ == '__main__':
    c = CardPredictor()
    c.load_svm()  # 加载训练好的模型
    r = c.predict(r"./result")
    print(r)
