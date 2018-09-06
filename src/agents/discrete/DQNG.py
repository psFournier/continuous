import numpy as np
RENDER_TRAIN = False
from networks import CriticDQNG
from agents import DQN
from buffers import ReplayBuffer

class DQNG(DQN):
    def __init__(self, args, env, env_test, logger):
        super(DQNG, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal']
        if args['--imit'] != '0':
            self.names.append('expVal')
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.critic = CriticDQNG(args, env)

    def train(self):
        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, g = [exp[name] for name in self.names]
            temp = np.expand_dims([1], axis=0)
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, g, temp])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1, g])
            targets_dqn = self.compute_targets(r, t, q)

            if self.args['--imit'] == '0':
                targets = targets_dqn
                inputs = [s0, a0, g]
            else:
                e = exp['expVal']
                targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
                inputs = [s0, a0, g, e]
            loss = self.critic.qvalModel.train_on_batch(inputs, targets)

            for i, metric in enumerate(self.critic.qvalModel.metrics_names):
                self.metrics[metric] += loss[i]

            self.critic.target_train()

    def make_input(self, state, t):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal]]
        # temp = self.env.explor_temp(t)
        input.append(np.expand_dims([0.5], axis=0))
        return input

    # def reset(self):
    #     super(DQNG, self).reset()
        # if self.args['--her'] != '0':
        #     self.hindsight()

    # def hindsight(self):
    #     reached = {goal: False for goal in self.env.goals if goal != self.env.goal}
    #     for goal, _ in reached.items():
    #         for expe in reversed(self.trajectory):
    #             r, term = self.env.eval_exp(expe, goal)
    #             if term:
    #                 reached[goal] = True
    #                 print('done goal {} unwillingly'.format(goal))
    #             if reached[goal]:
    #                 expe['goal'] = goal
    #                 expe['reward'] = r
    #                 expe['terminal'] = term
    #                 self.buffer.append(expe.copy())

    # def get_tutor_exp(self, goal):
    #     for i in range(10):
    #         state0 = self.env.reset()
    #         actions = rnd.choice(self.env.trajectories[goal])
    #         episode = []
    #         for a in actions:
    #             state1 = self.env.step(a)
    #             experience = {'state0': state0.copy(),
    #                           'action': a,
    #                           'state1': state1.copy(),
    #                           'reward': None,
    #                           'terminal': None,
    #                           'goal': None}
    #             state0 = state1
    #             episode.append(experience)
    #         self.process_tutor_episode(episode)
    #
    # def process_tutor_episode(self, episode):
    #     reached_goals = []
    #     for expe in reversed(episode):
    #         s0, a, s1 = expe['state0'], expe['action'], expe['state1']
    #         for goal in self.env.goals:
    #             is_new = goal not in reached_goals
    #             r, t = self.env.eval_exp(s0, a, s1, goal)
    #             if is_new and t:
    #                 reached_goals.append(goal)
    #             if not is_new or t:
    #                 new_expe = {'state0': s0,
    #                             'action': a,
    #                             'state1': s1,
    #                             'reward': r,
    #                             'terminal': t,
    #                             'goal': goal,
    #                             'R': None}
    #                 self.append_tutor_exp(new_expe)
    #
    # def append_tutor_exp(self, new_expe):
    #     pass
    #
    # def train_imitation(self):
    #     experiences = self.tutor_buffer.sample(self.batch_size)
    #     s0, a0, s1, r, t, e, step, g = [np.array(experiences[name]) for name in self.names]
    #
    #     targets_imit = np.zeros((self.batch_size, 1))
    #     self.critic.marginModel.train_on_batch(x=[s0, a0, g], y=targets_imit)
    #
    #     a1 = self.critic.actModel.predict_on_batch([s1, g])
    #     q = self.critic.qvalTModel.predict_on_batch([s1, a1, g])
    #     targets_dqn = self.compute_targets(r, t, q)
    #     self.critic.qvalModel.train_on_batch(x=[s0, a0, g], y=targets_dqn)
