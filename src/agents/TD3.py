import numpy as np
from networks import actorTD3
from networks import criticTD3
from agents.agent import Agent

RENDER_TRAIN = False
TARGET_CLIP = False
INVERTED_GRADIENTS = True

class TD3(Agent):

    def __init__(self, args, sess, env, logger, xy_sampler, eps_sampler, buffer):

        super(TD3, self).__init__(args, sess, env, logger, xy_sampler, eps_sampler, buffer)

        self.actor = actorTD3.ActorTD3(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         tau=0.005,
                                         learning_rate=0.001)

        self.critic = criticTD3.CriticTD3(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         gamma=0.99,
                                         tau=0.005,
                                         learning_rate=0.001)

    def train(self):
        critic_stats = []
        actor_stats = []
        experiences = self.env.buffer.sample(self.batch_size)
        td_errors, gradients = self.train_critic(experiences)
        if self.env.buffer.beta != 0:
            self.env.buffer.update_priorities(experiences['indices'], np.abs(td_errors[0]))
        if self.env_step % 2 == 0:
            self.train_actor(experiences, gradients)
            self.target_train()
        return np.array(critic_stats), np.array(actor_stats)

    def target_train(self):
        self.actor.target_train()
        self.critic.target_train()

    def train_actor(self, experiences, grads):

        a_outs = self.actor.model.predict_on_batch(experiences['state0'])
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

    def act(self, state):
        # action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)

        action = self.actor.model.predict(np.reshape(state, (1, self.actor.s_dim[0])))
        noise = np.random.normal(0., 0.1, size=action.shape)
        action = noise + action
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action

    def train_critic(self, experiences):

        # Calculate targets
        actions = self.actor.target_model.predict_on_batch(experiences['state1'])
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        noise = np.clip(np.random.normal(0., 0.2, size=actions.shape), -0.5, 0.5)
        actions = actions + noise
        target_q1 = self.critic.target_model1.predict_on_batch([experiences['state1'], actions])
        target_q2 = self.critic.target_model2.predict_on_batch([experiences['state1'], actions])
        target_q = np.minimum(target_q1, target_q2)
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
        td_errors, gradients = self.critic.train([experiences['state0'], experiences['action'], targets, experiences['weights']])
        self.critic.model2.train_on_batch([experiences['state0'], experiences['action']],
                                          targets)

        return td_errors, gradients