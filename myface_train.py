import cv2
import configparser
import numpy as np
import os
from PIL import Image

yIDs = []
xFaces = []
baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "dataset")

recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法


config = configparser.ConfigParser()
config.read('database.ini')
# faces = np.array(config['owner']['face'])

for root, dirs, files in os.walk(imageDir):
    print(root, dirs, files)
    for file in files:
        print(file)
        if file.endswith("png") or file.endswith("jpg"):
            # Retrieve USER ID from directory name
            path = os.path.join(root, file)
            id_ = int(os.path.basename(root))
            print("UID:" + str(id_))

            # Convert the face image to grayscale and convert pixel data to Numpy Array
            faceImage = Image.open(path).convert("L")
            faceArray = np.array(faceImage, "uint8")

            # Insert USER ID and face data into dataset
            yIDs.append(id_)
            xFaces.append(faceArray)

            # Display the face image to be used for training
            cv2.imshow("training", faceArray)
            cv2.waitKey(10)

recog.train(xFaces, np.array(yIDs))
recog.save("./training.yml")
print('ok')