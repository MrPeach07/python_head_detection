#coding=utf-8
import cv2


from numpy import *

cap = cv2.VideoCapture(0)

num = 0

face_cascade = cv2.CascadeClassifier(
    'I:/python/learning_code/pycharm_project/data/lbpcascade_frontalface.xml')


############################################

while True:
    # get a frame
    ret, img = cap.read()
    # show a frame
    cv2.imshow("capture", img)

    #######################检测人脸代码#####################
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度转换
    faces_rect = face_cascade.detectMultiScale(
        grayimg, 1.2, 3)  # 得到一个矩形，faces是多个矩形（一个脸一个）

    # 将多个脸用框画出来
    for (x, y, w, h) in faces_rect:
        img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi_gray = grayimg[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow('img', img)
    key = cv2.waitKey(10)
    cv2.imwrite('pics/%s.header.jpg' % (str(num)), img)
    num = num + 1
    ############################################

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

