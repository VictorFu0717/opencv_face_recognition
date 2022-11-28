import cv2
import numpy as np
import os
import configparser
import json

np.set_printoptions(threshold=np.inf)
detector = cv2.CascadeClassifier('/Users/victor/PycharmProjects/opencv_facerecognition-master/haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型

faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列



# Ask for the user's name
name = input("What's his/her Name?")


config = configparser.ConfigParser()
config.read('database.ini')
names = eval(config['owner']['name'])
number = len(names)+1
count = 1

def get_key(dict, value):       # 用value找key
    return [k for k, v in dict.items() if v == value][0]

if name in names.values():
    dirName = "./dataset"+"/"+ get_key(names, name)
else:
    dirName = "./dataset"+"/"+str(number)
    names[f'{number}'] = name
    config['owner']['name'] = str(names)
    with open('database.ini', 'w') as configfile:
        config.write(configfile)

if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("DataSet Directory Created")

print('camera...')                                # 提示啟用相機
cap = cv2.VideoCapture(0)                         # 啟用相機
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()                         # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    # img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列
    face = detector.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 4)
    for(x,y,w,h) in face:                         # 擷取人臉區域
        # faces.append(img_np[y:y+h,x:x+w])         # 記錄自己人臉的位置和大小內像素的數值
        fileName = dirName + "/" + f'{count}' + ".jpg"
        print(fileName)
        roi = gray[y:y+h,x:x+w]
        cv2.imwrite(fileName, roi)
        # cv2.imshow('face', roi)
        # ids.append(number)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
    cv2.imshow('oxxostudio', img)                      # 顯示攝影機畫面
    # print(faces)
    # print(ids)
    if count == 50:
        break
    if cv2.waitKey(100) == ord('q'):              # 每一毫秒更新一次，直到按下 q 結束
        break

# config['owner']['face'] = str(eval(config['owner']['face']) + faces)
# config['owner']['id'] = str(eval(config['owner']['id']) + ids)


