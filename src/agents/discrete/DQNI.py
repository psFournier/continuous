import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQNI
from agents.agent import Agent
from agents import DQN

from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNI(DQN):

    def __init__(self, args, env, env_test, logger):
        self.imitweight = float(args['--imitweight'])
        super(DQNI, self).__init__(args, env, env_test, logger)

    def init(self, env):
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.critic = CriticDQNI(s_dim=env.state_dim, num_a=env.action_dim, weight=self.imitweight)
        self.metrics['dqnloss'] = 0
        self.metrics['imitloss1'] = 0
        self.metrics['imitloss2'] = 0
        self.metrics['qval'] = 0

    def step(self):
        self.env_step += 1
        self.episode_step += 1
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, e = [np.array(experiences[name]) for name in self.names]
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1])
            targets_dqn = self.compute_targets(r, t, q)
            targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
            loss = self.critic.qvalModel.train_on_batch([s0, a0, e], targets)
            self.metrics['dqnloss'] += loss[1]
            self.metrics['imitloss1'] += loss[2]
            self.metrics['imitloss2'] += loss[3]
            self.metrics['qval'] += np.mean(loss[4])
            self.critic.target_train()