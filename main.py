from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import time
from tools import *
from trainers import *

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--gpu', type=int, help="gpu id", default=0)
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=2,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Breakout-v0',
    metavar='ENV',
    help='Target environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='Target environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load',
    default=False,
    metavar='L',
    help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--count-lives',
    default=False,
    metavar='CL',
    help='end of life is end of training episode.')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--use_convertor',
    default=False,
    type=bool,
    metavar='ENV',
    help='If should use the mapper')

# (Akshita) Arguments required for transfer learning.
parser.add_argument(
    '--model-env',
    default='Pong-v0',
    metavar='MENV',
    help='Source environment to train on (default: Pong-v0)')
parser.add_argument(
    '--model-env-config',
    default='config.json',
    metavar='MEC',
    help='Source environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--workers-transfer',
    type=int,
    default=2,
    metavar='WT',
    help='how many training processes to use (default: 2)')


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]

    if args.use_convertor:
        convertor_config = NetConfig('conversion_models/attention_breakout2pong_dual.yaml')
        hyperparameters = {}
        for key in convertor_config.hyperparameters:
            exec ('hyperparameters[\'%s\'] = convertor_config.hyperparameters[\'%s\']' % (key, key))

        #trainer = []
        #exec ("trainer=%s(convertor_config.hyperparameters)" % convertor_config.hyperparameters['trainer'])
        #trainer.gen.load_state_dict(torch.load('/home/spmunuku/Project/DRL/rl_a3c_pytorch/conversion_models/attentionbreakout2pong_v0_gen_00003500.pkl'))
        #trainer.gen.eval()
        #trainer.cuda(args.gpu)
        #trainer.share_memory()
        #distance_gan = trainer
        distance_gan = None
    else:
        convertor_config = None
        distance_gan = None
    convertor = distance_gan

    env = atari_env(args.env, env_conf, args, None, None, mapFrames=False)

    model_env = None
    if args.use_convertor:
        setup_json = read_config(args.model_env_config)
        model_env_conf = setup_json["Default"]
        for i in setup_json.keys():
            if i in args.model_env:
                model_env_conf = setup_json[i]
        model_env = atari_env(args.model_env, model_env_conf, args)


    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)


    # (TODO): We need to load the pretrained Pong weights so that the last layer (Ac-
    # tion spaces are different) does not get loaded.
    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    #blur_json = read_config(args.config_blur)
    #config_blur = blur_json["Default"]

    #p = mp.Process(target=test, args=(args, shared_model, env_conf, config_blur))
    p = mp.Process(target=test, args=(args, shared_model, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer, env_conf, None, None, None, False))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    if args.use_convertor:
        for rank1 in range(rank + 1, rank + 1 + args.workers_transfer):
            p = mp.Process(target=train, args=(
                rank1, args, shared_model, optimizer, env_conf, model_env_conf, convertor, convertor_config, True))
            p.start()
            processes.append(p)
            time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

