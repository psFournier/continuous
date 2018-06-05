import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
from ddpg.logger import Logger
import datetime
from ddpg.networks import ActorNetwork, CriticNetwork
from ddpg.ddpgAgent import DDPG_agent
from ddpg.util import load, boolean_flag
import json
import os
from ddpg.env_wrappers.base import Base
from ddpg.env_wrappers.registration import make

def main(args):
    params = [str(args['env']),
             str(args['her']),
             str(args['n_her_goals']),
             str(args['eps']),
             str(args['R']),
             str(args['beta'])]

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")

    # Different loggers to retrieve info at different moments in training.
    log_dir = os.path.join(args['log_dir'], '_'.join(params), now)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as config_file:
        config_file.write(json.dumps(args))
    logger_step = Logger(dir=os.path.join(log_dir,'log_steps'), format_strs=['stdout', 'json'])
    logger_episode = Logger(dir=os.path.join(log_dir,'log_episodes'), format_strs=['stdout', 'json'])

    # Make calls env_wrappers.registration.make, not the exact make function from the gym.
    env = make(args['env'])

    # Wrapper to override basic env methods and be able to access goal space properties. The wrapper classes paths corresponding to each environment are defined in ddpg.env_wrappers.init.
    if env.spec.wrapper_entry_point is not None:
        wrapper_cls = load(env.spec.wrapper_entry_point)
        env = wrapper_cls(env,
                                float(args['eps']),
                                int(args['R']),
                                float(args['beta']),
                                args['her'])
    else:
        env = Base(env, int(args['buffer_size']))


    with tf.Session() as sess:
        if args['random_seed'] is not None:
            np.random.seed(int(args['random_seed']))
            tf.set_random_seed(int(args['random_seed']))
            env.seed(int(args['random_seed']))

        actor = ActorNetwork(sess,
                             env.state_dim,
                             env.action_dim)

        critic = CriticNetwork(sess,
                               env.state_dim,
                               env.action_dim)

        agent = DDPG_agent(sess,
                           actor,
                           critic,
                           env,
                           logger_step,
                           logger_episode,
                           int(args['episode_steps']),
                           int(args['max_steps']),
                           int(args['eval_freq']))

        agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    parser.add_argument('--random-seed', help='random seed for repeatability', default=None)
    parser.add_argument('--env', help='choose the gym env', default='FetchReach_e-v1')
    parser.add_argument('--her', help='hindsight strategy', default='no_no')
    parser.add_argument('--n-her-goals', default=4)
    parser.add_argument('--n-split', help='number of split comparisons', default=10)
    parser.add_argument('--split-min', help='minimum cp difference to allow split', default=0.0001)
    parser.add_argument('--n-window', help='length of running window used to compute cp', default=3)
    parser.add_argument('--eps', default=0.02)
    parser.add_argument('--R', help='number of regions in goal space', default=4)
    parser.add_argument('--n-points', help='number of points stored in region', default=100)
    parser.add_argument('--beta', default=1)
    parser.add_argument('--max-steps', help='max num of episodes to do while training', default=500000)
    parser.add_argument('--log-dir', help='directory for storing run info',
                        default='/home/pierre/PycharmProjects/continuous/log/local/')
    parser.add_argument('--episode-steps', help='number of steps in the environment during evaluation', default=50)
    parser.add_argument('--eval-freq', help='freq for critic and actor stats computation', default=1000)

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
