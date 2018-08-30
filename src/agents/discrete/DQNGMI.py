import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGMI
from agents import DQN
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNGMI(DQN):
    def __init__(self, args, env, env_test, logger):
        super(DQNGMI, self).__init__(args, env, env_test, logger)

    def init(self, env):
        self.critic = CriticDQNGMI(s_dim=env.state_dim, g_dim=env.goal_dim, num_a=env.action_dim)
        self.names += ['expVal', 'goalVals', 'goal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.metrics['dqnloss'] = 0
        self.metrics['imitloss1'] = 0
        self.metrics['imitloss2'] = 0
        self.metrics['qval'] = 0

    def step(self):
        self.env_step += 1
        self.episode_step += 1
        self.env.steps[self.env.goal] += 1
        self.exp['goalVals'] = self.env.goalVals
        self.exp['goal'] = self.env.goal
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, e, gv, g = [np.array(experiences[name]) for name in self.names]
            m = np.array([self.env.obj2mask(g[k]) for k in range(self.batch_size)])
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, gv, m])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1, gv, m])
            targets_dqn = self.compute_targets(r, t, q)
            targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
            loss = self.critic.qvalModel.train_on_batch([s0, a0, gv, m, e], targets)
            self.metrics['dqnloss'] += loss[1]
            self.metrics['imitloss1'] += loss[2]
            self.metrics['imitloss2'] += loss[3]
            self.metrics['qval'] += np.mean(loss[4])
            self.critic.target_train()

    def make_input(self, state):
        return [np.expand_dims(i, axis=0) for i in [state, self.env.goalVals, self.env.mask]]

    def processEp(self):
        E = 0
        for expe in reversed(self.trajectory):
            if self.trajectory[-1]['terminal']:
                E = E * self.critic.gamma + expe['reward']
                expe['expVal'] = E
            else:
                expe['expVal'] = -self.ep_steps
            self.buffer.append(expe.copy())