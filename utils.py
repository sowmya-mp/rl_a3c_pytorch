from __future__ import division
import numpy as np
import torch
import json
import logging


def get_translator(args):
    return get_translator_from_source(args.model_env, args.env)

def get_translator_from_source(source, target):
    if 'DemonAttack' in target:
        translate = [0, 1, 1,2,3,0,0]
    elif 'Tennis' in target and 'Tennis' in source:
        translate = [0, 1, 0, 3, 4, 0, 3, 4, 3, 4, 1, 11, 12, 1, 11, 12, 11, 12]
    elif 'Assault' in target:
        translate = [0, 2, 3,4,3,4]
    elif 'Breakout' in target and 'Tennis' in source:
        translate = [0 ,1 , 0, 2, 3, 0, 2, 3, 2, 3, 1, 2, 3, 1 , 2, 3, 2, 3]
    elif 'Pong' in target and 'Tennis' in source:
        translate = [0 ,1 , 0, 2, 3, 0, 2, 3, 2, 3, 1, 4, 5, 1 , 4, 5, 4, 5]
    elif 'Pong' in target:
        translate = [0,1,4,5]
    elif 'Breakout' in target:
        translate = [0,1,2,3, 2, 3]
    elif 'Tennis' in target:
        translate = [0,1,3,4, 3, 4]
    else:
        raise Exception("Cant translate enviroment: {}".format(target))
    return translate

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        if not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

