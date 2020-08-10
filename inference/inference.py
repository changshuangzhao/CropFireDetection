import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
from time import time


def check_color(img1):
    '''
    hsv- green: 45<h<90  43<s<255  46<v<255
    yello: 5<h<20  43<s<255  46<v<255
    red: 0<h<20 or 175<h<255,
    '''
    img = img1.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # for red
    lower_red1 = np.array([0, 43, 43])
    upper_red1 = np.array([17, 255, 255])
    lower_red2 = np.array([180,43,43])
    upper_red2 = np.array([255,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    row, col, _ = img.shape
    if min(row, col) >= 1200:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    elif min(row, col) >= 900:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))
    elif min(row, col) >= 700:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    elif min(row, col) >= 500:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    elif min(row, col) >= 300:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    elif min(row, col) >= 150:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    else:
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    mask_img = cv2.dilate(mask, kernelX, iterations=2)
    mask_img = cv2.erode(mask_img, kernelX, iterations=4)
    mask_img = cv2.dilate(mask_img, kernelX, iterations=2)
    mask_img = cv2.dilate(mask_img, kernelY, iterations=2)
    mask_img = cv2.erode(mask_img, kernelY, iterations=1)
    mask_img = cv2.dilate(mask_img, kernelY, iterations=2)

    mask_img = cv2.medianBlur(mask_img, 3)

    _, contours, hier = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, c in enumerate(contours):
        # 边界框
        x, y, w, h = cv2.boundingRect(c)
        # 保存大于等于20的区域
        if min(w, h) >= 20:
            x1, x2, y1, y2 = int(x), int(x + w), int(y), int(y + h)
            boxes.append([x1, y1, x2, y2])
    return boxes


def inference():
    # 创建会话
    sess = tf.Session()
    # 加载pb模型
    with gfile.FastGFile('models/model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    # pb模型的输入：img_input，shape=(1, 3, 112, 112)
    img_input = sess.graph.get_tensor_by_name('img_input:0')
    # pb模型的输出：cls_out，预测的概率值，shape=(1, 2)
    cls_out = sess.graph.get_tensor_by_name('cls_out:0')

    # 图像路径
    image_path = '/Users/yanyan/data/FireData/TestData/Fire/fire2278.jpg'
    # 读取图像
    orig = cv2.imread(image_path)
    # 提取红色/黄色————可能存在火焰的区域
    boxes = check_color(orig)
    # 数据预处理
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    processed = transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(**imagenet_stats)])
    # 存储火焰区域
    fire_boxes = []
    for box in boxes:
        det_img = orig[box[1]: box[3], box[0]: box[2]]
        det_img = Image.fromarray(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        det_img = processed(det_img)
        det_img = torch.unsqueeze(det_img, dim=0)
        det_img = det_img.numpy()
        tic = time()
        pro = sess.run(cls_out, feed_dict={img_input: det_img})
        toc = time()
        print('Time Taken = ', toc - tic)
        if np.argmax(pro) == 0:
            fire_boxes.append(box)
    # 绘制火焰区域
    if fire_boxes:
        for fire_box in fire_boxes:
            x1, y1, x2, y2 = fire_box
            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('fire detection', orig)
        cv2.waitKey()


if __name__ == '__main__':
    inference()

