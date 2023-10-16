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

def homorphic_filter(img):
    ### homomorphic filter는 gray scale image에 대해서 밖에 안 되므로

    ### YUV color space로 converting한 뒤 Y에 대해 연산을 진행

    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    

    y = img_YUV[:,:,0]    

    

    rows = y.shape[0]    

    cols = y.shape[1]

 

### illumination elements와 reflectance elements를 분리하기 위해 log를 취함

    imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)

    

    ### frequency를 이미지로 나타내면 4분면에 대칭적으로 나타나므로 

    ### 4분면 중 하나에 이미지를 대응시키기 위해 row와 column을 2배씩 늘려줌

    M = 2*rows + 1

    N = 2*cols + 1

    

    ### gaussian mask 생성 sigma = 10

    sigma = 10

    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬

    Xc = np.ceil(N/2) # 올림 연산

    Yc = np.ceil(M/2)

    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성

    

    ### low pass filter와 high pass filter 생성

    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))

    HPF = 1 - LPF

    

    ### LPF랑 HPF를 0이 가운데로 오도록iFFT함. 

    ### 사실 이 부분이 잘 이해가 안 가는데 plt로 이미지를 띄워보니 shuffling을 수행한 효과가 났음

    ### 에너지를 각 귀퉁이로 모아 줌

    LPF_shift = np.fft.ifftshift(LPF.copy())

    HPF_shift = np.fft.ifftshift(HPF.copy())

    

    ### Log를 씌운 이미지를 FFT해서 LPF와 HPF를 곱해 LF성분과 HF성분을 나눔

    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))

    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분

    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

    

    ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함

    gamma1 = 0.3

    gamma2 = 1.5

    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]

    

    ### 조정된 데이터를 이제 exp 연산을 통해 이미지로 만들어줌

    img_exp = np.expm1(img_adjusting) # exp(x) + 1

    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화

    img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌

    

    ### 마지막으로 YUV에서 Y space를 filtering된 이미지로 교체해주고 RGB space로 converting

    img_YUV[:,:,0] = img_out

    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return result


class mirnet:
    def __init__(self, size):
        self.model = tf.keras.models.load_model("photon3", compile=False)
        self.desired_size = size
    def infer(self, image):
        orginal_image = self.preprocessing(image)
        curr= time.time()
        enhanced_image =  self.model.predict(orginal_image)
        print(f"image inference time : {time.time()-curr}")
        return self.postprocessing(enhanced_image)

    def preprocessing(self, image):
        curr = time.time()
        original_image = cv2.resize(image, dsize=self.desired_size, interpolation=cv2.INTER_LINEAR)
        img = keras.utils.img_to_array(original_image)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)    
        print(f"image preprocessing time : {time.time()-curr}")
        return img
    
    def postprocessing(self, image):
        curr= time.time()
        output = image[0].reshape(
        (self.desired_size[0], self.desired_size[1], 3)
    )
        output_image = output * 255.0
        output_image = output_image.clip(0, 255)
        image = Image.fromarray(np.uint8(output_image))
        print(f"image postprocessing time : {time.time()-curr}")

        return np.uint8(image) 

class dce:
    def __init__(self, size):
        self.model = tf.keras.models.load_model("dce2", compile=False)
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

    model = mirnet(size=(256, 256))

    rtsp = "rtsp://210.99.70.120:1935/live/cctv001.stream"
    # 공공데이터 충청남도 천안시_교통정보 CCTV RTSP 샤나인코더
    capture = cv2.VideoCapture('testvideo2.mp4') # 노트북의 경우 0, 외부 장치 번호가 1~n 까지 순차적으로 할당

    # 카메라의 속성 설정 메서드 set
    # capture.set(propid, value)로 카메라의 속성(propid)과 값(value)을 설정
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    count = 0

    # while을 통해서 카메라에서 프레임을 지속적으로 받는다.
    while cv2.waitKey(33) < 0:
        # ret = 카메라 상태, 비정상이면 False
        # frame = 현재 시점의 프레임
        ret, frame = None, None
        for _ in range(3):
            time.sleep(0.01)
            ret, frame = capture.read()
        # flip : flipcode 가 0 이면 가로대칭 변경. 1이면 세로대칭 변경 
       
        rst_img = model.infer(frame) 
        #input size (512, 512)
        curr = time.time()
        #rst_img = homorphic_filter(rst_img)
        print(f"image filtering = {time.time() - curr}" )
        resized_img = cv2.resize(frame, dsize=model.desired_size, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("VideoFrame", cv2.hconcat([resized_img, rst_img]))
        
    
    # 카메라 장치에서 받아온 메모리 해제
    capture.release()
    # 모든 윈도우 창 제거
    cv2.destroyAllWindows()
    