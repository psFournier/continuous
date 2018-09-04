from gym import Wrapper
import numpy as np
import math
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule

# Wrappers override step, reset functions, as well as the defintion of action, observation and goal spaces.

class Rnd(Wrapper):
    def __init__(self, env, args):
        super(Rnd, self).__init__(env)
        self.shaping = args['--shaping'] != '0'
        self.opt_init = int(args['--opt_init'])
        self.goal = None
        self.init()

    def init(self):
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.interests = [0 for _ in self.goals]
        self.dones = [0 for _ in self.goals]
        self.explorations = [LinearSchedule(schedule_timesteps=int(5000),
                                            initial_p=1.0,
                                            final_p=.5) for _ in self.goals]

    def processEp(self, episode):
        T = int(episode[-1]['terminal'])
        if T:
            print('done')
            self.dones[self.goal] += 1
        R = np.sum([exp['reward'] for exp in episode])
        S = len(episode)
        self.queues[self.goal].append({'R': R, 'S': S, 'T': T})

    def step(self, action):
        return self.env.step(action)

    def is_term(self, exp):
        return False

    def eval_exp(self, exp):
        term = self.is_term(exp)
        if term:
            r = 1
        else:
            r = 0
        r = self.transform_r(r, term)
        exp['reward'] = r
        exp['terminal'] = term
        return exp

    def transform_r(self, r, term):
        if self.opt_init == 1:
            r += self.gamma - 1
            if term:
                r -= self.gamma
        elif self.opt_init == 2:
            r += self.gamma - 1
        elif self.opt_init == 3:
            r -= 1
        return r

    def get_idx(self):
        CPs = [abs(q.CP) for q in self.queues]
        maxCP = max(CPs)
        minCP = min(CPs)

        Rs = [q.R for q in self.queues]
        maxR = max(Rs)
        minR = min(Rs)

        try:
            if maxCP > 1:
                self.interests = [(cp - minCP) / (maxCP - minCP) for cp in CPs]
            else:
                self.interests = [1 - (r - minR) / (maxR - minR) for r in Rs]
        except:
            self.interests = [1 for _ in self.queues]

        weighted_interests = [math.pow(I, self.theta) + 0.1 for I in self.interests]
        sum = np.sum(weighted_interests)
        mass = np.random.random() * sum
        idx = 0
        s = weighted_interests[0]
        while mass > s:
            idx += 1
            s += weighted_interests[idx]
        return idx

    def get_stats(self):
        stats = {}
        for i, goal in enumerate(self.goals):
            stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].R))
            stats['CP_{}'.format(goal)] = float("{0:.3f}".format(abs(self.queues[i].CP)))
            stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[i]))
            stats['S_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].S))
            stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].T))
            stats['done_{}'.format(goal)] = float("{0:.3f}".format(self.dones[i]))
            stats['step_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
        return stats



    # @property
    # def min_avg_length_ep(self):
    #     return np.min([q.L_mean for q in self.queues if q.L_mean != 0])