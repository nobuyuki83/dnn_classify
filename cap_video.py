import os
from datetime import datetime

import cv2

def capture_camera(mirror=True):
    cap = cv2.VideoCapture(1) 

    while True:
        ret, frame = cap.read() # ret is a frag if capture is successfull

        if mirror is True:
            frame = frame[:,::-1]

        frame2 = frame[::2,::2]
        cv2.imshow('camera capture', frame2)

        dt = datetime.now()
        str_dt = dt.strftime('%Y%m%d_%H%M%S')

        key = cv2.waitKey(1) # 1msec待つ
        if key == 27: 
            break
        if key == 49: # 1 karintou shiro
            cv2.imwrite('img_train/1/'+str_dt+'.png',frame2)
        if key == 50: # 2 plane buiscuit
            cv2.imwrite('img_train/2/'+str_dt+'.png',frame2)
        print(key)

    cap.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.isdir('img_train/1'):
        os.mkdir('img_train/1')
    if not os.path.isdir('img_train/2'):
        os.mkdir('img_train/2')

    capture_camera(mirror=False)