import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
from agents import DQN, DQNG, TD3, DDPG2, DQNGM, Qoff
from utils.logger import Logger
import datetime
from utils.util import load
import json
import os
from env_wrappers.registration import make
import gym.spaces
from docopt import docopt

help = """

Usage: 
  main.py --env=<ENV> --agent=<AGENT> [options]

Options:
  --seed SEED              Random seed
  --theta THETA            CP importance for goal selection [default: 0]
  --shaping YES_NO         Reward shaping [default: 0]
  --opt_init YES_NO        Positive initialisation [default: 0]
  --max_steps VAL          Maximum total steps [default: 200000]
  --ep_steps VAL           Maximum episode steps [default: 200]
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/continuous/log/local/]
  --eval_freq VAL          Logging frequency [default: 2000]
  --her YES_NO             Hindsight Experience Replay [default: 0]
  --margin VAL             Large margin loss margin [default: 0.1]
  --gamma VAL              Discount factor [default: 0.99]
  --batchsize VAL          Batch size [default: 64]
  --wimit VAL              Weight for imitaiton loss with imitaiton [default: 0]
  --filter VAL             Type of filter [default: 1]
"""

def build_logger(args):
    param_strings = [args['--agent'], args['--env']]
    now = datetime.datetime.now()
    log_dir = os.path.join(args['--log_dir'], '_'.join(param_strings), now.strftime("%Y%m%d%H%M%S_%f"))
    os.makedirs(log_dir, exist_ok=True)
    args['--time'] = now
    print(args)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as config_file:
        config_file.write(json.dumps(args, default=str))
    logger = Logger(dir=os.path.join(log_dir,'log_steps'),
                         format_strs=['stdout', 'json', 'tensorboard_{}'.format(int(args['--eval_freq']))])

    return logger

if __name__ == '__main__':
    args = docopt(help)
    logger = build_logger(args)
    env = make(args['--env'], args)
    env_test = make(args['--env'], args)
    seed = args['--seed']
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)
        env_test.seed(seed)

    # if args['agent'] == 'ddpg':
    #     agent = DDPG(args, env, env_test, logger)
    # elif args['agent'] == 'td3':
    #     agent = TD3(args, env, env_test, logger)
    if args['--agent'] == 'qoff':
        agent = Qoff(args, env, env_test, logger)
    elif args['--agent'] == 'dqn':
        agent = DQN(args, env, env_test, logger)
    elif args['--agent'] == 'dqng':
        agent = DQNG(args, env, env_test, logger)
    elif args['--agent'] == 'dqngm':
        agent = DQNGM(args, env, env_test, logger)
    elif args['--agent'] == 'ddpg2':
        agent = DDPG2(args, env, env_test, logger)
    else:
        raise RuntimeError
    # elif args['agent'] == 'qlearning':
    #     agent = Qlearning.Qlearning(args, sess, env, env_test, logger)
    # elif args['agent'] == 'qlearning_off':
    #     agent = Qlearning_offpolicy.Qlearning_offPolicy(args, sess, env, env_test, logger)
    # elif args['agent'] == 'qlearningfd':
    #     with open(os.path.join(args['log_dir'],
    #                            'qlearning_Taxi-v1_1',
    #                            '20180628141516_051865',
    #                            'policy.pkl'), 'rb') as input:
    #         Q_tutor = pickle.load(input)
    #     agent = QlearningfD.QlearningfD(args, sess, env, env_test, Q_tutor, logger)

    agent.run()
