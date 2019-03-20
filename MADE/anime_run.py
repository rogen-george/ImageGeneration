"""
Trains MADE on Anime Faces Dataset MNIST, which can be downloaded here:
https://drive.google.com/file/d/1jdJXkQIWVGOeb0XJIXE3YuZQeiEPd8rM/view?usp=sharing.
"""
import argparse
import torchvision

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from made import MADE

def run_epoch(split, upto=None):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else samples
    x = xtr if split == 'train' else xte
    N,D = x.size()
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % resample_every == 0 or split == 'test': # if in test, cycle masks every time
                #print("!! nsamples = ", nsamples)
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        loss = F.mse_loss(xbhat, xb, size_average=False) / B
        lossf = loss.data.item()
        lossfs.append(lossf)

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

    print("%s epoch average loss: %f" % (split, np.mean(lossfs)))

# This method takes 25 sample anime faces from the data and saves it into a png file
def plot_anime_faces(data):

    padding1 = np.ones((5,64)) * 255
    padding2 = np.ones((64,5)) * 255
    axis_num = 0
    padding = padding1
    images = []
    images_group = []
    for i in range(5):
        for j in range(5):
            images.append(data[j+i].reshape(64, 64))
        test = torch.cat(images)
        images_group.append(test)
        images = []
    grid = torch.cat(images_group, 1)
    plt.imsave('anime_sample.png', grid.reshape(320,320).cpu().numpy(), cmap='Greys_r')



def run_epoch_test(split, upto=None):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else samples
    x = xtr if split == 'train' else xte
    N,D = x.size()
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % resample_every == 0 or split == 'test': # if in test, cycle masks every time
                #print("!! nsamples = ", nsamples)
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        #get 25 samples to save to a png file
        print (xbhat.shape)
        samp = xbhat[0:25,:]

        samp_matrix = (samp.data).cpu().numpy()

        # evaluate the binary cross entropy loss
        loss = F.mse_loss(xbhat, xb, size_average=False) / B
        lossf = loss.data.item()
        lossfs.append(lossf)

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

    print("%s epoch average loss: %f" % (split, np.mean(lossfs)))
    return (samp)


from matplotlib.pyplot import cm
from matplotlib.pyplot import imsave

if __name__ == '__main__':

    num_masks = 1

    # Number of Hidden Layers and Hidden Units
    hiddens = "8000,8000,8000"
    samples = 64
    resample_every = 1

    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    flag = True

    # Location of the pickle file
    pickle_file = "anime.pkl"
    pickle_file = open(pickle_file, 'rb')
    try:
        for i in range(1000):
            imgarray = pickle.load(pickle_file)
            imgarray = imgarray.flatten()
            imgarray = imgarray.reshape([1, imgarray.shape[0]])
            if flag:
                spcinv_data = imgarray
                flag = False
            else:
                spcinv_data = np.vstack((spcinv_data, imgarray))
    except EOFError:
        pass

    print("shape", spcinv_data.shape)

    test_data = torch.from_numpy(spcinv_data)
    print("type", type(test_data))
    pickle.dump(test_data,open('anime_np.txt', 'wb'))
    print("type", type(spcinv_data))
    train_split_index = spcinv_data.shape[0]
    train_split_index = int(train_split_index * 4 / 5)
    xtr = spcinv_data[:train_split_index,:]
    xte = spcinv_data[train_split_index:,:]
    print("xtr.shape: ", xtr.shape)
    print("xte.shape: ", xte.shape)
    xtr = torch.from_numpy(xtr).cuda()
    print("xtr_type: ", xtr)
    xte = torch.from_numpy(xte).cuda()

    xtr = xtr.float()
    xte = xte.float()

    # construct model and ship to GPU
    hidden_list = list(map(int, hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1), num_masks=num_masks)
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

    # start the training
    for epoch in range(100):
        scheduler.step(epoch)
        run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
        run_epoch('train')

    print("optimization done. full test set eval:")
    samp = run_epoch_test('test')
    plot_anime_faces(samp)
