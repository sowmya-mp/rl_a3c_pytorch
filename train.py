from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
from transfer_util import frame2attention
from utils import get_translator_from_source


def train(rank, args, shared_model, optimizer, env_conf, model_env_conf=None, convertor=None, convertor_config=None, mapFrames=False):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    # TODO(Akshita): Change this to make the environments as required.
    num_of_actions = 4

    if args.use_convertor:
        #convertor_config = NetConfig('conversion_models/attention_breakout2pong_dual.yaml')
        #hyperparameters = {}
        #for key in convertor_config.hyperparameters:
        #    exec ('hyperparameters[\'%s\'] = convertor_config.hyperparameters[\'%s\']' % (key, key))

        trainer = []
        exec ("trainer=%s(convertor_config.hyperparameters)" % convertor_config.hyperparameters['trainer'])
        trainer.gen.load_state_dict(torch.load('/home/spmunuku/Project/DRL/rl_a3c_pytorch/conversion_models/attentionbreakout2pong_v0_gen_00003500.pkl'))
        trainer.gen.eval()
        #trainer.cuda(args.gpu)
        distance_gan = trainer
    else:
        convertor_config = None
        distance_gan = None
    convertor = distance_gan

    if mapFrames:
        env = atari_env("{}".format(args.model_env), model_env_conf, args, convertor, convertor_config, mapFrames)
        # env_id = args.model_env
        num_of_actions = atari_env("{}".format(args.env), env_conf, args).action_space
    else:
        env = atari_env("{}".format(args.env), env_conf, args)
        # env_id = args.env
        num_of_actions = env.action_space

    # print("num_of_actions", num_of_actions)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)



    player = Agent(None, env, args, None)

    # (Akshita): Get the action translator.
    if mapFrames:
        player.translator = get_translator_from_source(args.env, args.model_env)
        player.translate_test = True

    player.gpu_id = gpu_id
    player.model = A3Clstm(
        player.env.observation_space.shape[0], num_of_actions)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)


        # print("num_steps", args.num_steps)
        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            if player.info['ale.lives'] == 0 or player.max_length:
                player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 100.0)
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()

