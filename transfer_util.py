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
    frame = frame[config["crop1"]:config["crop2"] + 160, :160]
    old_frame = frame
    orig_ata = frame
    dilation = frame

    frame = cv2.resize(frame, (256, 256))
    cv2.imwrite('frames/original.jpg', frame)
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

    cv2.imwrite('frames/blurred.jpg', frame)
    return frame, orig_ata, old_frame, dilation
