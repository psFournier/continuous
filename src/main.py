import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
from agents import DQN, DDPG, TD3
from agents import Qlearning, Qlearning_offpolicy
from utils.logger import Logger
import datetime
from utils.util import load
import json
import os
from env_wrappers.registration import make
import gym.spaces

def build_logger(args):
    params = ['agent', 'env', 'theta']
    param_strings = [str(args[name]) for name in params]
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    log_dir = os.path.join(args['log_dir'], '_'.join(param_strings), now)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as config_file:
        config_file.write(json.dumps(args))
    logger = Logger(dir=os.path.join(log_dir,'log_steps'),
                         format_strs=['stdout', 'json', 'tensorboard_{}'.format(args['eval_freq'])])
    # logger = Logger(dir=os.path.join(log_dir, 'log_steps'), format_strs=['stdout', 'json'])

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for agent')

    parser.add_argument('--random-seed', default=None)
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--agent', default='td3')
    # parser.add_argument('--her_xy', default='no')
    # parser.add_argument('--her_eps', default='no')
    # parser.add_argument('--n_split', default=10)
    # parser.add_argument('--split_min', default=0.0001)
    # parser.add_argument('--window', default=100)
    # parser.add_argument('--alpha', default=0)
    # parser.add_argument('--beta', default=0.4)
    # parser.add_argument('--eps', default=0.02)
    # parser.add_argument('--queue_len', default=200)
    parser.add_argument('--theta', default=1)

    # parser.add_argument('--R', help='must be power of 2', default=128)
    parser.add_argument('--max_steps', help='max num of episodes to do while training', default=1000000)
    parser.add_argument('--log_dir', help='directory for storing run info',
                        default='/home/pierre/PycharmProjects/continuous/log/local/')
    parser.add_argument('--episode_steps', help='number of steps in the environment during evaluation', default=200)
    parser.add_argument('--eval_freq', help='freq for critic and actor stats computation', default=5000)

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    logger = build_logger(args)
    env = make(args['env'], args)
    env_test = make(args['env'], args)

    if args['random_seed'] is not None:
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
        env_test.seed(int(args['random_seed']))

    with tf.Session() as sess:

        if args['agent'] == 'ddpg':
            agent = DDPG.DDPG(args, sess, env, env_test, logger)
        elif args['agent'] == 'td3':
            agent = TD3.TD3(args, sess, env, env_test, logger)
        elif args['agent'] == 'dqn':
            agent = DQN.DQN(args, sess, env, env_test, logger)
        elif args['agent'] == 'qlearning':
            agent = Qlearning.Qlearning(args, sess, env, env_test, logger)
        elif args['agent'] == 'qlearning_off':
            env_tutor = make(args['env'], args)
            agent = Qlearning_offpolicy.Qlearning_offPolicy(args, sess, env, env_test, env_tutor, logger)
        else:
            raise RuntimeError

        agent.run()
