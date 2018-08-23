import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
from agents import DQN, DQNG, TD3, DDPG, DQNGM
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
  --self_imit YES_NO       Imitates itself [default: 0]
  --tutor_imit YES_NO      Imitates the tutor [default: 0]
  --seed SEED              Random seed
  --theta THETA            CP importance for goal selection [default: 0]
  --beta BETA              CP importance for training samples weights [default: 0]
  --shaping YES_NO         Reward shaping [default: 0]
  --posInit YES_NO         Positive initialisation [default: 0]
  --max_steps VAL          Maximum total steps [default: 200000]
  --ep_steps VAL           Maximum episode steps [default: 200]
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/continuous/log/local/]
  --eval_freq VAL          Logging frequency [default: 2000]
  --per YES_NO             Prioritized Experience Replay [default: 0]
  --her YES_NO             Hindsight Experience Replay [default: 0]

"""

def build_logger(args):
    param_strings = [args['--agent'], args['--env']]
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    log_dir = os.path.join(args['--log_dir'], '_'.join(param_strings), now)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as config_file:
        config_file.write(json.dumps(args))
    logger = Logger(dir=os.path.join(log_dir,'log_steps'),
                         format_strs=['stdout', 'json', 'tensorboard_{}'.format(int(args['--eval_freq']))])

    return logger

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='provide arguments for agent')
    #
    # parser.add_argument('--random-seed', default=1)
    # parser.add_argument('--env', default='HalfCheetah-v2')
    # parser.add_argument('--agent', default='td3')
    # parser.add_argument('--per', default=0)
    # parser.add_argument('--her', default=0)
    # parser.add_argument('--self_imit', default=0)
    # parser.add_argument('--tutor_imit', default=0)
    # parser.add_argument('--theta', default=0)
    # parser.add_argument('--beta', default=0)
    # parser.add_argument('--shaping', default=0)
    # parser.add_argument('--posInit', default=0)
    #
    # parser.add_argument('--max_steps', help='max num of episodes to do while training', default=500000)
    # parser.add_argument('--log_dir', help='directory for storing run info',
    #                     default='/home/pierre/PycharmProjects/continuous/log/local/')
    # parser.add_argument('--episode_steps', help='number of steps in the environment during evaluation', default=200)
    # parser.add_argument('--eval_freq', help='freq for critic and actor stats computation', default=2000)
    #
    # args = vars(parser.parse_args())
    #
    # pp.pprint(args)

    args = docopt(help)
    print(args)

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
    if args['--agent'] == 'dqn':
        agent = DQN(args, env, env_test, logger)
    elif args['--agent'] == 'dqng':
        agent = DQNG(args, env, env_test, logger)
    elif args['--agent'] == 'dqngm':
        agent = DQNGM(args, env, env_test, logger)
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
