
import cv2
import os, numpy, math, hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


def get_image_tnsr_from_img_path(path_img, npix):
    assert os.path.isfile(path_img)
    img1 = cv2.imread(path_img)
    ####
    img3 = img1.copy()
    h3 = img3.shape[0]
    w3 = img3.shape[1]
    nbh3 = math.ceil(h3 / npix)
    nbw3 = math.ceil(w3 / npix)
    img3 = cv2.copyMakeBorder(img3, 0, npix * nbh3 - h3, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img3 = cv2.copyMakeBorder(img3, 0, 0, 0, npix * nbw3 - w3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ####
    tnsr = numpy.moveaxis(img3, 2, 0).astype(numpy.float32) / 255.0
    ####
    return img3, tnsr


# Get N image and rectanble from aPath
def get_batch(list_path_class, npix, nblock):
    np_batch_in = []
    np_batch_trg_c = []
    nelem = len(list_path_class)
    for i in range(nelem):
        assert len(list_path_class[i]) == 2
        pth0 = list_path_class[i][0]
        cls0 = list_path_class[i][1]-1
        img, tnsr = get_image_tnsr_from_img_path(pth0, npix)
        ###
        np_batch_in.append(tnsr.flatten())
        np_batch_trg_c.append(cls0)
    np_batch_in = numpy.asarray(np_batch_in).reshape(nelem, 3, npix * nblock[0], npix * nblock[1])
    np_batch_trg_c = numpy.asarray(np_batch_trg_c).reshape(nelem)
    return np_batch_in, np_batch_trg_c

###################################################################

def load_network_classification(is_cuda, path_dir):
    ####
    net_d = NetDesc()
    if os.path.isfile(path_dir + '/model_d.pt'):
        if is_cuda:
            net_d.load_state_dict(torch.load(path_dir + '/model_d.pt'))
        else:
            net_d.load_state_dict(torch.load(path_dir + '/model_d.pt', map_location='cpu'))

    ####
    net_c = NetClass()
    if os.path.isfile(path_dir + '/model_c.pt'):
        if is_cuda:
            net_c.load_state_dict(torch.load(path_dir + '/model_c.pt'))
        else:
            net_c.load_state_dict(torch.load(path_dir + '/model_c.pt', map_location='cpu'))

    if is_cuda:
        net_d = net_d.cuda()
        net_c = net_c.cuda()

    return net_d, net_c


def pad32TrchV(trchVImg0):
    npix = 32
    h0 = trchVImg0.size()[1]
    w0 = trchVImg0.size()[2]
    H0 = math.ceil(h0 / npix)
    W0 = math.ceil(w0 / npix)
    trchV = F.pad(trchVImg0, (0, W0 * npix - w0, 0, H0 * npix - h0))
    return trchV


def detect_face_dnn(net_d, net_c, net_l, trchVImg, threshold_prob):
    ###
    npix = net_d.npix
    is_cuda = next(net_d.parameters()).is_cuda
    assert next(net_c.parameters()).is_cuda == is_cuda
    assert next(net_l.parameters()).is_cuda == is_cuda
    sm = nn.Softmax(dim=1)
    ###
    trchVImg2 = trchVImg.view(1, 3, trchVImg.shape[1], trchVImg.shape[2])
    #	print(input_img2.shape,input_img.shape)
    descriptor = net_d(trchVImg2)
    nblockH = descriptor.shape[2]
    nblockW = descriptor.shape[3]
    assert trchVImg.shape[1] == nblockH * npix
    assert trchVImg.shape[2] == nblockW * npix
    #	print(descriptor.shape,nblockH,nblockW)
    output_c = net_c(descriptor)
    output_c = sm(output_c)
    output_l = net_l(descriptor)
    if is_cuda:
        np_desc = descriptor.cpu().data.numpy()
        np_outc = output_c.cpu().data.numpy()
        np_outl = output_l.cpu().data.numpy()
    else:
        np_desc = descriptor.data.numpy()
        np_outc = output_c.data.numpy()
        np_outl = output_l.data.numpy()
    #	print(descriptor.shape, output_c.shape, output_l.shape)

    aRect = []
    for ih in range(nblockH):
        for iw in range(nblockW):
            rect0 = [npix * iw, npix * ih, npix, npix]
            if np_outc[0, 1, ih, iw] < threshold_prob: continue
            dx = np_outl[0, 0, ih, iw]
            dy = np_outl[0, 1, ih, iw]
            ds = np_outl[0, 2, ih, iw]
            r2 = math.pow(2, ds * 2) * npix
            r0 = (dx + iw + 0.5) * npix - r2 * 0.5
            r1 = (dy + ih + 0.5) * npix - r2 * 0.5
            #            print(dx,dy,ds,"  ",r0,r1,r2, math.pow(2,ds*2),npix)
            rect1 = [r0, r1, r2, r2]
            aRect.append(rect1)
    return aRect


def detect_face_dnn_multires(net_d, net_c, net_l, npImg, threshold_prob, nblk_max=15):
    is_cuda = next(net_d.parameters()).is_cuda
    assert next(net_c.parameters()).is_cuda == is_cuda
    assert next(net_l.parameters()).is_cuda == is_cuda
    net_d.eval()
    net_c.eval()
    net_l.eval()
    convNp2Tensor = torchvision.transforms.ToTensor()
    trchVImg0 = Variable(convNp2Tensor(npImg), requires_grad=False)
    if is_cuda:
        trchVImg0 = trchVImg0.cuda()

    ####
    list_rect = []
    #    trchVImg0 = trchVImg0[:, ::1, ::1]
    trchVImg0a = pad32TrchV(trchVImg0)
    if max(trchVImg0a.shape[1:3]) / 32 < nblk_max and min(trchVImg0a.shape[1:3]) / 32 > 2:
        list_rect0 = detect_face_dnn(net_d, net_c, net_l, trchVImg0a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect0, 1.0)
    ####
    trchVImg1 = trchVImg0[:, ::2, ::2]
    trchVImg1a = pad32TrchV(trchVImg1)
    if max(trchVImg1a.shape[1:3]) / 32 < nblk_max and min(trchVImg1a.shape[1:3]) / 32 > 2:
        list_rect1 = detect_face_dnn(net_d, net_c, net_l, trchVImg1a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect1, 2.0)
    ####
    trchVImg2 = trchVImg0[:, ::4, ::4]
    trchVImg2a = pad32TrchV(trchVImg2)
    if max(trchVImg2a.shape[1:3]) / 32 < nblk_max and min(trchVImg2a.shape[1:3]) / 32 > 2:
        list_rect2 = detect_face_dnn(net_d, net_c, net_l, trchVImg2a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect2, 4.0)
    ####
    trchVImg3 = trchVImg0[:, ::8, ::8]
    trchVImg3a = pad32TrchV(trchVImg3)
    if max(trchVImg3a.shape[1:3]) / 32 < nblk_max and min(trchVImg3a.shape[1:3]) / 32 > 2:
        list_rect3 = detect_face_dnn(net_d, net_c, net_l, trchVImg3a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect3, 8.0)
    return list_rect


def detect_face_dnn_multires_merge(net_d, net_c, net_l, npImg):
    list_rect = detect_face_dnn_multires(net_d, net_c, net_l, npImg)
    list_rect_merge = marge_rects(list_rect)
    return list_rect_merge


def initialize_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, 1.414)
            nn.init.constant(m.bias, 0.1)
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class NetClass(nn.Module):
    def __init__(self):
        super(NetClass, self).__init__()
        self.nchanel_in = 256
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
        #		self.bn1   = nn.BatchNorm2d(64)
        ###
        self.conv2 = nn.Conv2d(64, 16, kernel_size=1)
        #		self.bn2    = nn.BatchNorm2d(16)
        ####
        self.conv3 = nn.Conv2d(16, 2, kernel_size=1)
        #		self.bn3    = nn.BatchNorm2d(2)
        ###
        initialize_net(self)

    def forward(self, x):
        #		x = F.relu(self.bn1(self.conv1(x)))
        #		x = F.relu(self.bn2(self.conv2(x)))
        #		x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class NetUnit(nn.Module):
    def __init__(self, nc):
        super(NetUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(nc)
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(nc)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        initialize_net(self)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


class NetDesc(nn.Module):
    def __init__(self):
        super(NetDesc, self).__init__()
        ####
        self.conv1a = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)  # 1/2
        ####
        self.unit2a = NetUnit(64)
        self.unit2b = NetUnit(64)
        self.unit2c = NetUnit(64)
        self.bn2 = nn.BatchNorm2d(64)
        ####
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # 1/2
        self.unit3a = NetUnit(128)
        self.unit3b = NetUnit(128)
        self.unit3c = NetUnit(128)
        self.bn3 = nn.BatchNorm2d(128)
        ####
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # 1/2
        self.unit4a = NetUnit(256)
        self.unit4b = NetUnit(256)
        self.unit4c = NetUnit(256)
        self.bn4 = nn.BatchNorm2d(256)
        ####
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # 1/2
        self.unit5a = NetUnit(256)
        self.unit5b = NetUnit(256)
        self.unit5c = NetUnit(256)
        self.bn5 = nn.BatchNorm2d(256)
        ####
        self.npix = 32
        self.nchanel_out = 256
        ####
        initialize_net(self)

    def forward(self, x):
        x = self.conv1a(x) # 1/2 * 64
        ####
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 1/4
        x = self.unit2a(x)
        x = self.unit2b(x)
        x = self.unit2c(x)
        x = F.relu(self.bn2(x))
        #####
        x = self.conv3(x) # 1/8 * 128
        x = self.unit3a(x)
        x = self.unit3b(x)
        x = self.unit3c(x)
        x = F.relu(self.bn3(x))
        #####
        x = self.conv4(x) # 1/16 * 256
        x = self.unit4a(x)
        x = self.unit4b(x)
        x = self.unit4c(x)
        x = F.relu(self.bn4(x))
        #####
        x = self.conv5(x) # 1/32 * 256
        x = self.unit5a(x)
        x = self.unit5b(x)
        x = self.unit5c(x)
        x = F.relu(self.bn5(x))
        return x
