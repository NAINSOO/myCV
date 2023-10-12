import cv2
from PIL import Image
import matplotlib.pyplot as plt
import keras
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from glob import glob
import time
import cv2
import multiprocessing

class dce:
    def __init__(self, model:str, size):
        self.model = tf.keras.models.load_model(model, compile=False)
        self.desired_size = size
    def infer(self, image):
        orginal_image = self.preprocessing(image)
        curr= time.time()
        enhanced_image =  self.model.predict(orginal_image, batch_size=4)
        print(f"image inference time : {time.time()-curr}")
        return self.postprocessing(enhanced_image)

    def preprocessing(self, image):
        curr = time.time()

        original_image = cv2.resize(image, dsize=self.desired_size, interpolation=cv2.INTER_LINEAR)

        original_image_part = []
        original_image_part.append(original_image[0:256, 0:256].copy())
        original_image_part.append(original_image[256:512, 0:256].copy())
        original_image_part.append(original_image[0:256, 256:512].copy())
        original_image_part.append(original_image[256:512, 256:512].copy())

        imgs = []
        for image in original_image_part:
            img = keras.utils.img_to_array(image)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            imgs.append(img)
        image = np.concatenate(imgs,axis=0)
        print(f"image preprocessing time : {time.time()-curr}")
        return image
    
    def postprocessing(self, images):
        curr= time.time()
        output_image = images[0]
        output_image = np.array(output_image) * 255
        output_image = output_image.clip(0, 255)
    
        output_images = []
        for img in output_image:
            image = Image.fromarray(np.uint8(img))
            output_images.append(np.uint8(image))
        #output_image = Image.fromarray(np.uint8(output_image))
        print(f"image postprocessing time : {time.time()-curr}")

        vertical_img1 = cv2.vconcat([output_images[0], output_images[1]])
        vertical_img2 = cv2.vconcat([output_images[2], output_images[3]])

        rst_img = cv2.hconcat([vertical_img1, vertical_img2])
        return rst_img



if __name__=='__main__':

    model = dce('dce2',size=(512, 512))

    rtsp = "rtsp://210.99.70.120:1935/live/cctv002.stream"
    # 공공데이터 충청남도 천안시_교통정보 CCTV RTSP
    capture = cv2.VideoCapture('testvideo.mp4') # 노트북의 경우 0, 외부 장치 번호가 1~n 까지 순차적으로 할당

    # 카메라의 속성 설정 메서드 set
    # capture.set(propid, value)로 카메라의 속성(propid)과 값(value)을 설정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

    count = 0

    # while을 통해서 카메라에서 프레임을 지속적으로 받는다.
    while cv2.waitKey(10) < 0:
        # ret = 카메라 상태, 비정상이면 False
        # frame = 현재 시점의 프레임
        ret, frame = capture.read()
        # flip : flipcode 가 0 이면 가로대칭 변경. 1이면 세로대칭 변경 
        
        rst_img = model.infer(frame) 
        #input size (512, 512)

        cv2.imshow("VideoFrame", cv2.hconcat([cv2.resize(frame, dsize=model.desired_size, interpolation=cv2.INTER_LINEAR), rst_img]))
        
    
    # 카메라 장치에서 받아온 메모리 해제
    capture.release()
    # 모든 윈도우 창 제거
    cv2.destroyAllWindows()
    