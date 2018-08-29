import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents import DQNG
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNGM(DQNG):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM, self).__init__(args, env, env_test, logger)

    def init(self, env):
        self.critic = CriticDQNGM(s_dim=env.state_dim, g_dim=env.goal_dim, num_a=env.action_dim)
        self.names += ['goalVals', 'goal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.metrics['dqnloss'] = 0
        self.metrics['qval'] = 0

    def step(self):
        self.env_step += 1
        self.episode_step += 1
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
            targets = self.compute_targets(r, t, q)
            # weights = np.array([self.env.interests[obj] ** self.beta for obj in g])
            loss = self.critic.qvalModel.train_on_batch([s0, a0, gv, m], targets)
            self.metrics['dqnloss'] += loss[0]
            self.metrics['qval'] += np.mean(loss[1])
            self.critic.target_train()

    def make_input(self, state):
        return [np.expand_dims(i, axis=0) for i in [state, self.env.goalVals, self.env.mask]]