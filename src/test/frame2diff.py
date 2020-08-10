import numpy as np 
import cv2


def check_color(img1):
    '''
    hsv- green: 45<h<90  43<s<255  46<v<255
    yello: 5<h<20  43<s<255  46<v<255
    red: 0<h<20 or 175<h<255,
    '''
    img = img1.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # for red
    lower_blue1 = np.array([0, 43, 43])
    upper_blue1 = np.array([17, 255, 255])
    lower_blue2 = np.array([180,43,43])
    upper_blue2 = np.array([255,255,255])
    # # Threshold the HSV image to get only blue colors
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    # cv2.imshow('mask1', mask1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    # cv2.imshow('mask2', mask2)
    mask = mask1 + mask2
    # cv2.imshow('mask', mask)

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
    # cv2.imshow('mask_img1', mask_img)
    mask_img = cv2.erode(mask_img, kernelX, iterations=4)
    # cv2.imshow('mask_img2', mask_img)
    mask_img = cv2.dilate(mask_img, kernelX, iterations=2)
    # cv2.imshow('mask_img3', mask_img)
    mask_img = cv2.dilate(mask_img, kernelY, iterations=2)
    # cv2.imshow('mask_img4', mask_img)
    mask_img = cv2.erode(mask_img, kernelY, iterations=1)
    mask_img = cv2.dilate(mask_img, kernelY, iterations=2)

    mask_img = cv2.medianBlur(mask_img, 3)
    final_mask = cv2.bitwise_and(img, img, mask=mask_img)
    cv2.imshow('final_mask', final_mask)

    _, contours, hier = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # hull = []
    for i, c in enumerate(contours):
        # 边界框
        x, y, w, h = cv2.boundingRect(c)
        # hull.append(cv2.convexHull(c, False))
        if min(w, h) >= 20:
            x1, x2, y1, y2 = int(x), int(x + w), int(y), int(y + h)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            boxes.append([x1, y1, x2, y2])
        # cv2.drawContours(img, hull, i, (0, 0, 255), 1, 8)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    return boxes


if __name__ == '__main__':
    import os
    # fpath = '/Users/yanyan/data/FireData/TestData/Fire/fire1258.jpg'
    # img = cv2.imread(fpath)
    # check_color(img)

    root = '/Users/yanyan/data/FireData/TrainData/Fire'
    img_names = os.listdir(root)
    for img_name in img_names:
        img_path = os.path.join(root, img_name)
        img = cv2.imread(img_path)
        check_color(img)