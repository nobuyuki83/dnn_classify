import os
from datetime import datetime

import cv2

def capture_camera(mirror=True, size=None):
    cap = cv2.VideoCapture(1) 

    while True:
        ret, frame = cap.read() # ret is a frag if capture is successfull

        if mirror is True:
            frame = frame[:,::-1]

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        cv2.imshow('camera capture', frame)

        dt = datetime.now()
        str_dt = dt.strftime('%Y%m%d_%H%M%S')

        key = cv2.waitKey(1) # 1msec待つ
        if key == 27: 
            break
        if key == 49: #1
            cv2.imwrite('img_train/1/'+str_dt+'.png',frame)
        if key == 50: #2
            cv2.imwrite('img_train/2/'+str_dt+'.png',frame)
        print(key)

    cap.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.isdir('img_train/1'):
        os.mkdir('img_train/1')
    if not os.path.isdir('img_train/2'):
        os.mkdir('img_train/2')

    capture_camera(mirror=False)