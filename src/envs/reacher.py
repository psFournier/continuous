import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(os.path.dirname(__file__), 'reacher.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, None, None, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
        position :
        cos(theta1), cos(theta2), sin(theta1), sin(theta2), theta1', theta2'
        fingertip_x, fingertip_y

        total size : 8
        """
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:2],
            # self.get_body_com("fingertip")[:2]
            self.get_body_com("fingertip")[:2] - self.get_body_com("target")[:2]
        ])

class ReacherDEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(os.path.dirname(__file__), 'reacherDeceptive.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, None, None, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        # qpos[-2:] = np.array([0, 0.1])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
        position :
        cos(theta1), cos(theta2), sin(theta1), sin(theta2), theta1', theta2'
        fingertip_x, fingertip_y

        total size : 8
        """
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:2],
            # self.get_body_com("fingertip")[:2]
            self.get_body_com("fingertip")[:2] - self.get_body_com("target")[:2]
        ])