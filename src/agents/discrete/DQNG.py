import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNG
from agents import DQN
from buffers import ReplayBuffer, PrioritizedReplayBuffer
import random as rnd


class DQNG(DQN):
    def __init__(self, args, env, env_test, logger):
        super(DQNG, self).__init__(args, env, env_test, logger)
        self.beta = float(args['--beta'])

    def init(self, env):
        self.critic = CriticDQNG(s_dim=env.state_dim, g_dim=env.goal_dim, num_a=env.action_dim)
        self.names += ['goal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.metrics['dqnloss'] = 0
        self.metrics['qval'] = 0

    def step(self):
        self.env_step += 1
        self.episode_step += 1
        self.env.steps[self.env.goal] += 1
        self.exp['goal'] = self.env.goal
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, e, g = [np.array(experiences[name]) for name in self.names]
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, g])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1, g])
            targets_dqn = self.compute_targets(r, t, q)
            # weights = np.array([self.env.interests[gi] ** self.beta for gi in g])
            loss = self.critic.qvalModel.train_on_batch([s0, a0, g], targets_dqn)
            self.metrics['dqnloss'] += loss[0]
            self.metrics['qval'] += np.mean(loss[1])
            self.critic.target_train()

    def make_input(self, state):
        return [np.expand_dims(i, axis=0) for i in [state, self.env.goal]]

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
