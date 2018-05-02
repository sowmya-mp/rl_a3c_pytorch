from __future__ import division, print_function
import gym
import os
import numpy as np
from gym.spaces.box import Box
from skimage.color import rgb2gray
import json
import cv2
import logging
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import itertools
from torch.autograd import Variable
import random
import cv2
from collections import deque

def dist_exp(im):
    e_im = np.exp(-im/6.)
    e_im = 1.0-e_im
    new_im = e_im*255.0
    return new_im.astype(np.uint8)

def frame2attention(frame, config, environment):
    #print(frame.shape)
    frame = np.transpose(frame,[1,2,0])
    frame = frame[config["crop1"]:config["crop2"] + 160, :160]
    old_frame = frame
    orig_ata = frame
    dilation = frame

    #print(frame.shape)
    frame = cv2.resize(frame, (256, 256))
    #cv2.imwrite('frames/original.jpg', frame)
    frame = rgb2gray(frame) * 255.
    orig_ata = frame

    if 'Pong' in environment:
        median_im = 87.
        frame = np.absolute((frame - median_im)).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)

    _, filtered_bin_image = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(filtered_bin_image, kernel, iterations=1)

    inv_im = 255 - dilation
    inv_im_dist = cv2.distanceTransform(inv_im, cv2.DIST_L2, 3)
    exp_inv_im_dist = dist_exp(inv_im_dist)
    exp_im_dist = 255 - exp_inv_im_dist

    exp_inv_im_dist = dist_exp(inv_im_dist / 2)
    exp_im_dist_1 = 255 - exp_inv_im_dist

    frame = np.stack((dilation, exp_im_dist_1, exp_im_dist), axis=-1)

    if 'Pong' in environment:
        frame = np.transpose(frame, [1, 0, 2])

    #cv2.imwrite('frames/blurred.jpg', frame)
    return frame, orig_ata, old_frame, dilation

def create_xy_image(width=256):
    coordinates = list(itertools.product(range(width), range(width)))
    arr = (np.reshape(np.asarray(coordinates), newshape=[width, width, 2]) - width/2 ) / float((width/2))
    new_map = np.transpose(np.float32(arr), [2, 0, 1])
    xy = torch.from_numpy(new_map).clone()
    xy = xy.unsqueeze(0).expand(1, xy.size(0), xy.size(1), xy.size(2))
    return xy

def attention_process_frame(frame, gan_trainer, gan_config):

    blurr = False
    rotate_A = gan_config.datasets['train_a']['rotation']
    rotate_B = gan_config.datasets['train_b']['rotation']
    cols = frame.shape[1]
    rows = frame.shape[0]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_A, 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))
    frame = frame.transpose(2, 0, 1)
    if not blurr:
        frame = frame[-1:, :, :]
    final_data = torch.from_numpy((frame / 255.0 - 0.5) * 2).float().clone()
    final_data = final_data.contiguous()
    final_data = final_data.resize_(1, final_data.size(0), final_data.size(1), final_data.size(2))

    #Use xy flag
    xy = create_xy_image()
    final_data = torch.cat([final_data, xy], 1)
    final_data_in = Variable(final_data.cuda())
    final_data_in = final_data_in.contiguous()

    output_data = gan_trainer.gen.forward_a2b(final_data_in)

    output_img = output_data[0].data.cpu().numpy()
    new_output_img = np.transpose(output_img, [2, 3, 1, 0])
    new_output_img = new_output_img[:, :, :, 0]
    out_img = np.uint8(255 * (new_output_img / 2 + 0.5))
    frame = out_img
    cols = frame.shape[1]
    rows = frame.shape[0]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360 - rotate_B, 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))

    if frame.shape[-1] != 3:
        frame = np.stack((frame, frame, frame), axis=-1)

    return frame