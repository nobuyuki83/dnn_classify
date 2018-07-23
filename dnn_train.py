import glob, random, math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import dnn_util


def size_block(size, mag):
    nbh2 = math.ceil(size[0] / mag)
    nbw2 = math.ceil(size[1] / mag)
    h = math.ceil(nbh2 / 32)
    w = math.ceil(nbw2 / 32)
    return (h, w)


class TrainingData:
    def __init__(self,batch_size):
        list1 = glob.glob('img_train/1/*.png')
        list1 = list(map(lambda path: (path,1,1), list1))

        list2 = glob.glob('img_train/2/*.png')
        list2 = list(map(lambda path: (path,2,1), list2))

        self.list12 = list1+list2
        print("class1: "+str(len(list1))+"  "+"class2: "+str(len(list2)))

        random.shuffle(self.list12)
        print(self.list12)

        self.iepoch = 0
        self.batch_size = batch_size
        self.ibatch = 0

    def get_batch(self):
        nbatch = int(math.ceil(len(self.list12)/self.batch_size))
        iend = (self.ibatch+1)*self.batch_size if self.ibatch<nbatch-1 else len(self.list12)

        list_path_class_batch = self.list12[self.ibatch*self.batch_size:iend]
        self.ibatch = self.ibatch+1
        if self.ibatch >= nbatch:
            self.ibatch = 0
            self.iepoch = self.iepoch+1
            random.shuffle(self.list12)

        return list_path_class_batch


def train(net_d, net_c, is_cuda, nitr, is_d, is_c,
          training_data):

    assert isinstance(is_d, bool)
    assert isinstance(is_c, bool)
    assert net_c.nchanel_in == net_d.nchanel_out

    difference_c = nn.CrossEntropyLoss(ignore_index=-1)

    param = []
    if is_d: param = list(net_d.parameters()) + param
    if is_c: param = list(net_c.parameters()) + param
    optimizer = optim.Adam(param, lr=0.0001)

    if is_d:
        net_d.train()
    else:
        net_d.eval()
    if is_c:
        net_c.train()
    else:
        net_c.eval()

    for itr in range(nitr):
#        nblock = (13,10)
        nblock = (10,8)
        npix = 32
        list_path_class = training_data.get_batch()
        np_in, np_trgc = dnn_util.get_batch(list_path_class, npix, nblock)
        input = Variable(torch.from_numpy(np_in), requires_grad=True)
        target_c = Variable(torch.from_numpy(np_trgc), requires_grad=False)
        if is_cuda:
            input = input.cuda()
            target_c = target_c.cuda()
        ###
        optimizer.zero_grad()
        descriptor = net_d(input)
        average_descriptor = torch.nn.functional.max_pool2d(descriptor, kernel_size=descriptor.size()[2:])
        output_c = net_c(average_descriptor)
        output_c = output_c.view(-1,2)
        loss_c = difference_c(output_c, target_c)
        ###
        loss = loss_c
        ####
        print(training_data.iepoch, "/", training_data.ibatch, nblock,"/", loss.data[0])
        loss.backward()
        optimizer.step()

    if is_d: torch.save(net_d.state_dict(), 'model_d.pt')
    if is_c: torch.save(net_c.state_dict(), 'model_c.pt')


def main():

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("using GPU")
    else:
        print("using CPU")

    net_d, net_c = dnn_util.load_network_classification(is_cuda, '.')

    training_data = TrainingData(10)

    for i in range(20):
        train(net_d,net_c,is_cuda,100,True,True,training_data)


if __name__ == "__main__":
    main()
