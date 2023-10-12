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

def infer(model, image):
    curr = time.time()
    img = keras.utils.img_to_array(image)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)    
    #image =  np.concatenate(imgs,axis=0)
    print(f"image preprocessing time : {time.time()-curr}")
    curr = time.time()
    output = None
    with tf.device('/gpu:0'):
        output = model.predict(img)
    print(f"image inference time : {time.time()-curr}")
    curr = time.time()

    output = output[0].reshape(
        (np.shape(image)[0], np.shape(image)[1], 3)
    )
    output_image = output * 255.0
    output_image = output_image.clip(0, 255)
    image = Image.fromarray(np.uint8(output_image))
    print(f"image postprocessing time : {time.time()-curr}")

    return np.uint8(image)
    

def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([600, 400, 3])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image


if __name__=='__main__':
    print("hello")

    desired_size = (512, 512)

    model = tf.keras.models.load_model('dce', compile=False)


    # 공공데이터 충청남도 천안시_교통정보 CCTV RTSP

    risp_address= 'rtsp://210.99.70.120:1935/live/cctv002.stream'

    capture = cv2.VideoCapture('testvideo.mp4') # 노트북의 경우 0, 외부 장치 번호가 1~n 까q지 순차적으로 할당

    # 카메라의 속성 설정 메서드 set
    # capture.set(propid, value)로 카메라의 속성(propid)과 값(value)을 설정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    count = 0

    # while을 통해서 카메라에서 프레임을 지속적으로 받는다.
    while cv2.waitKey(20) < 0:
        # ret = 카메라 상태, 비정상이면 False
        # frame = 현재 시점의 프레임
        ret, frame = capture.read()        
        
        if ret:
            original_image = cv2.resize(frame, dsize=desired_size, interpolation=cv2.INTER_LINEAR)
            image = infer(model, original_image)
            cv2.imshow("VideoFrame", cv2.hconcat([original_image, image]))
        
    
    # 카메라 장치에서 받아온 메모리 해제
    capture.release()
    # 모든 윈도우 창 제거
    cv2.destroyAllWindows()
    