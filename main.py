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


def infer(model, original_image):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    original_image = Image.fromarray(np.uint8(original_image))
    return output_image, np.uint8(output_image)


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([600, 400, 3])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image


if __name__=='__main__':
    print("hello")
    pool = multiprocessing.Pool(4)

    desired_size = (600, 400)
    new_model = tf.keras.models.load_model('mirnet', compile=False)

    # 공공데이터 충청남도 천안시_교통정보 CCTV RTSP
    capture = cv2.VideoCapture('rtsp://210.99.70.120:1935/live/cctv001.stream') # 노트북의 경우 0, 외부 장치 번호가 1~n 까지 순차적으로 할당

    # 카메라의 속성 설정 메서드 set
    # capture.set(propid, value)로 카메라의 속성(propid)과 값(value)을 설정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    count = 0 

    # while을 통해서 카메라에서 프레임을 지속적으로 받는다.
    while cv2.waitKey(33) < 0:
        # ret = 카메라 상태, 비정상이면 False
        # frame = 현재 시점의 프레임
        ret, frame = capture.read()
        # flip : flipcode 가 0 이면 가로대칭 변경. 1이면 세로대칭 변경 
        
        original_image = cv2.resize(frame, dsize=desired_size, interpolation=cv2.INTER_LINEAR)
        #_, enhanced_image = infer(new_model, original_image)
        image_part = []
        for i in range(4):
            _, image_part[i] = pool.apply(infer, (new_model, frame))

            pool.close()
            pool.join()
        cv2.imshow("VideoFrame", frame)
        #cv2.imshow("VideoFrame", enhanced_image)
    
    # 카메라 장치에서 받아온 메모리 해제
    capture.release()
    # 모든 윈도우 창 제거
    cv2.destroyAllWindows()
    