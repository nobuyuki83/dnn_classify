import os,math,numpy
from datetime import datetime

import cv2

import torch
from torch.autograd import Variable

import dnn_util

def classify(img3):
    npix = 32
    h3 = img3.shape[0]
    w3 = img3.shape[1]
    nbh3 = math.ceil(h3 / npix)
    nbw3 = math.ceil(w3 / npix)
    img3 = cv2.copyMakeBorder(img3, 0, npix * nbh3 - h3, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img3 = cv2.copyMakeBorder(img3, 0, 0, 0, npix * nbw3 - w3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    npTnsrImg = numpy.moveaxis(img3, 2, 0).astype(numpy.float32) / 255.0
    ####
    trchVImg = Variable(torch.from_numpy(npTnsrImg), requires_grad=False)
    trchVImg.contiguous()
    trchVImg2 = trchVImg.view(1, 3, trchVImg.shape[1], trchVImg.shape[2])
    descriptor = net_d(trchVImg2)
    average_descriptor = torch.nn.functional.max_pool2d(descriptor, kernel_size=descriptor.size()[2:])
    output_c = net_c(average_descriptor)
    output_c = output_c.view(2)
    return output_c

def capture_camera(net_d, net_c, mirror=True):
    assert net_c.nchanel_in == net_d.nchanel_out
    net_c.eval()
    net_d.eval()
    is_cuda = next(net_d.parameters()).is_cuda
    assert next(net_c.parameters()).is_cuda == is_cuda

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read() # ret is a frag if capture is successfull

        if mirror is True:
            frame = frame[:,::-1]

        frame2 = frame[0::2,0::2].copy()
        output_c = classify(frame2)
        soft_max = torch.nn.Softmax(dim=0)
        output_c1 = soft_max(output_c)

        prob1 = output_c1.data[0]
        prob2 = output_c1.data[1]

        cv2.line(frame2, (10,10),(10+int(prob1*100),10),(255,255,255), thickness=5)
        cv2.line(frame2, (10,20),(10+int(prob2*100),20),(255,0,0), thickness=5)

        cv2.imshow('camera capture', frame2)

        dt = datetime.now()
        str_dt = dt.strftime('%Y%m%d_%H%M%S')

        key = cv2.waitKey(1) # wait 1 millisec
        if key == 27:
            break

    cap.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.isdir('img_train'):
        os.mkdir('img_train')
    if not os.path.isdir('img_train/1'):
        os.mkdir('img_train/1')
    if not os.path.isdir('img_train/2'):
        os.mkdir('img_train/2')

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("using GPU")
    else:
        print("using CPU")

    net_d, net_c = dnn_util.load_network_classification(is_cuda, '.')

    capture_camera(net_d, net_c, mirror=False)