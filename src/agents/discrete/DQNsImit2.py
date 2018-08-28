import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQNsImit2
from agents.agent import Agent
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNsImit2(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DQNsImit2, self).__init__(args, env, env_test, logger)
        self.per = args['--per'] != '0'
        self.self_imitation = args['--self_imit'] != '0'
        self.tutor_imitation = args['--tutor_imit'] != '0'
        self.her = args['--her'] != '0'
        self.trajectory = []
        self.margin = float(args['--margin'])
        self.imitweight = float(args['--imitweight'])
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'expVal']
        self.init(env)

    def init(self, env):
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.bufferImit = ReplayBuffer(limit=int(1e6), names=self.names)
        self.critic = CriticDQNsImit2(s_dim=env.state_dim,
                                num_a=env.action_dim,
                                gamma=0.99,
                                tau=0.001,
                                learning_rate=0.001,
                                margin=self.margin,
                                imit=self.imitweight)
        self.metrics['imitloss'] = 0

    def step(self):
        self.env_step += 1
        self.episode_step += 1
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, e = [np.array(experiences[name]) for name in self.names]
            a1 = self.critic.actModel.predict_on_batch([s1])
            q = self.critic.qvalTModel.predict_on_batch([s1, a1])
            targets_dqn = self.compute_targets(r, t, q)
            dqnL, qval = self.critic.qvalModel.train_on_batch(x=[s0, a0], y=targets_dqn)
            self.metrics['dqnloss'] += dqnL
            self.metrics['qval'] += np.mean(qval)
            self.critic.target_train()

        if self.bufferImit.nb_entries > self.batch_size:
            experiencesImit = self.bufferImit.sample(self.batch_size)
            s0, a0, s1, r, t, e = [np.array(experiencesImit[name]) for name in self.names]
            a1 = self.critic.actModel.predict_on_batch([s1])
            q = self.critic.qvalTModel.predict_on_batch([s1, a1])
            targets_dqn = self.compute_targets(r, t, q)
            targets_imit = np.zeros((self.batch_size, 1))
            imitL, margin = self.critic.marginModel.train_on_batch(x=[s0, a0, e],
                                                                   y=[targets_imit, targets_dqn])
            self.metrics['imitloss'] += imitL
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
            R, E, S = 0, 0, 0
            if self.trajectory[-1]['terminal']:
                print('done')
                self.env.dones[self.env.goal] += 1
            for expe in reversed(self.trajectory):
                R += int(expe['reward'])
                S += 1
                if self.trajectory[-1]['terminal']:
                    E = E * self.critic.gamma + expe['reward']
                    expe['expVal'] = E
                    self.bufferImit.append(expe)
                else:
                    expe['expVal'] = -self.ep_steps
                self.buffer.append(expe)
            self.env.queues[self.env.goal].append({'step': self.env_step, 'R': R, 'S': S})
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def make_input(self, state):
        return [np.reshape(state, (1, self.critic.s_dim[0]))]

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.env.explorations[self.env.goal].value(self.env_step):
            action = np.random.randint(0, self.env.action_dim)
        else:
            input = self.make_input(state)
            action = self.critic.actModel.predict(input, batch_size=1)
            action = action[0, 0]
        return action
