from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from transfer_util import frame2attention


def test(args, shared_model, env_conf, convertor=None, convertor_config=None, mapFrames=False):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args, convertor, convertor_config, mapFrames)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(
        player.env.observation_space.shape[0], player.env.action_space)

    player.state = player.env.reset()
    #player.state, _, _, _ = frame2attention(player.state, blur_conf, str(env.spec))
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and player.info['ale.lives'] > 0 and not player.max_length:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.done or player.max_length:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            print("Save", args.save_max, reward_sum, max_score)
            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        print('{0}{1}.dat'.format(args.save_model_dir, args.env))
                        torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

