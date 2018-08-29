import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQN
from agents.agent import Agent
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQN(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DQN, self).__init__(args, env, env_test, logger)
        self.per = args['--per'] != '0'
        self.her = args['--her'] != '0'
        self.trajectory = []
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal']
        self.init(env)

    def init(self, env):
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.critic = CriticDQN(s_dim=env.state_dim, num_a=env.action_dim)
        self.metrics['dqnloss'] = 0
        self.metrics['qval'] = 0

    def step(self):
        self.env_step += 1
        self.episode_step += 1
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t = [np.array(experiences[name]) for name in self.names]
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1])
            targets_dqn = self.compute_targets(r, t, q)
            loss = self.critic.qvalModel.train_on_batch([s0, a0], targets_dqn)
            self.metrics['dqnloss'] += loss[0]
            self.metrics['qval'] += np.mean(loss[1])
            self.critic.target_train()

    def compute_targets(self, r, t, q, clip=True):
        targets = []
        for k in range(self.batch_size):
            target = r[k] + (1 - t[k]) * self.critic.gamma * q[k]
            if clip:
                target_clip = np.clip(target, -0.99 / (1 - self.critic.gamma), 0.01)
                targets.append(target_clip)
            else:
                targets.append(target)
        targets = np.array(targets)
        return targets

    def reset(self):

        if self.trajectory:
            if self.trajectory[-1]['terminal']:
                print('done')
                self.env.dones[self.env.goal] += 1
            R = np.sum([exp['reward'] for exp in self.trajectory])
            S = len(self.trajectory)
            self.env.queues[self.env.goal].append({'step': self.env_step, 'R': R, 'S': S})
            self.processEp()
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def processEp(self):
        for expe in reversed(self.trajectory):
            self.buffer.append(expe.copy())

    def make_input(self, state):
        return [np.reshape(state, (1, self.critic.s_dim[0]))]

    def act(self, state, noise=False):

        if noise and np.random.rand(1) < self.env.explorations[self.env.goal].value(self.env_step):
            action = np.random.randint(0, self.env.action_dim)
        else:
            input = self.make_input(state)
            actionProbs = self.critic.actionProbsModel.predict(input, batch_size=1)
            # action = np.random.choice(range(self.env.action_dim), p=actionProbs[0])
            action = np.argmax(actionProbs[0], axis=0)
        return action
