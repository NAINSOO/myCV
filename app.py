from flask import Flask
import cv2
from flask import Response
from main import mirnet
import time
from flask import render_template

app = Flask(__name__)
model = mirnet(size=(256, 256))



def get_frame():
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
        frame = cv2.hconcat([resized_img, rst_img])
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
    # 카메라 장치에서 받아온 메모리 해제
    capture.release()
    
@app.route('/')
def Index():
    return render_template('template.html')  # index.html 파일을 렌더링하여 반환합니다.

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0')