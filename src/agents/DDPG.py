import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import actorDdpg, criticDdpg
from agents.agent import Agent
import pickle
from buffers.replayBuffer import ReplayBuffer
from buffers.prioritizedReplayBuffer import PrioritizedReplayBuffer

class DDPG(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(DDPG, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = ReplayBuffer(limit=int(1e6),
                                       names=['state0', 'action', 'state1', 'reward', 'terminal'])

        self.actor = actorDdpg.ActorDDPG(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         tau=0.001,
                                         learning_rate=0.0001)

        self.critic = criticDdpg.CriticDDPG(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         gamma=0.99,
                                         tau=0.001,
                                         learning_rate=0.001)

    def train(self, exp):

        self.env.buffer.append(exp)

        critic_stats = []
        actor_stats = []

        if self.env_step > 3 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size)
            td_errors = self.train_critic(experiences)
            if self.env.buffer.beta != 0:
                self.env.buffer.update_priorities(experiences['indices'], np.abs(td_errors[0]))
            self.train_actor(experiences)
            self.target_train()

        return np.array(critic_stats), np.array(actor_stats)

    def target_train(self):
        self.actor.target_train()
        self.critic.target_train()

    def init_targets(self):
        self.actor.hard_target_train()
        self.critic.hard_target_train()

    def train_critic(self, experiences):

        actions = self.actor.target_model.predict_on_batch(experiences['state1'])
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        target_q = self.critic.target_model.predict_on_batch([experiences['state1'], actions])
        y_i = []
        for k in range(self.batch_size):
            if TARGET_CLIP:
                target_q[k] = np.clip(target_q[k],
                                      self.env.reward_range[0] / (1 - self.critic.gamma),
                                      self.env.reward_range[1])

            if experiences['terminal'][k]:
                y_i.append(experiences['reward'][k])
            else:
                y_i.append(experiences['reward'][k] + self.critic.gamma * target_q[k])

        targets = np.reshape(y_i, (self.batch_size, 1))
        td_errors = self.critic.train([experiences['state0'],
                                                  experiences['action'],
                                                  targets,
                                                  experiences['weights']])
        return td_errors

    def train_actor(self, experiences):

        a_outs = self.actor.model.predict_on_batch(experiences['state0'])
        q_vals, grads = self.critic.gradients(experiences['state0'], a_outs)
        if INVERTED_GRADIENTS:
            """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
            low = self.env.action_space.low
            high = self.env.action_space.high
            for d in range(grads[0].shape[0]):
                width = high[d]-low[d]
                for k in range(self.batch_size):
                    if grads[k][d]>=0:
                        grads[k][d] *= (high[d]-a_outs[k][d])/width
                    else:
                        grads[k][d] *= (a_outs[k][d]-low[d])/width
        stats = self.actor.train(experiences['state0'], grads)
        return stats

    def act(self, state, noise=True):

        action = self.actor.model.predict(np.reshape(state, (1, self.actor.s_dim[0])))
        if noise:
            noise = np.random.normal(0., 0.1, size=action.shape)
            action = noise + action
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = action.squeeze()

        return action

    def log(self):
        if self.env_step % self.eval_freq == 0:
            returns = []
            for _ in range(10):
                state = self.env_test.reset()
                r = 0
                terminal = False
                step = 0
                while (not terminal and step < self.ep_steps):
                    action = self.act(state, noise=False)
                    experience = self.env_test.step(action)
                    r += experience['reward']
                    terminal = experience['terminal']
                    state = experience['state1']
                    step += 1
                returns.append(r)
            self.stats['avg_return'] = np.mean(returns)
            self.stats['step'] = self.env_step
            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])
            self.logger.dumpkvs()

    def save_regions(self):
        with open(os.path.join(self.log_dir, 'regionTree.pkl'), 'wb') as output:
            pickle.dump(self.env.regionTree, output)
