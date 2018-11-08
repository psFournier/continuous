import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import six


class ManipulatorEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(os.path.dirname(__file__), 'manipulator.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']
        # self.target = 'target'
        self.sites = ['grasp']


    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, None, None, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1 * self.model.stat.extent
        self.viewer.cam.lookat[2] = 10
        self.viewer.cam.camid = 0
        self.viewer.cam.fovy = 4

    def reset_model(self):

        # Find a collision-free random initial configuration.
        penetrating = True

        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        body_pos = self.model.body_pos.copy()
        body_quat = self.model.body_quat.copy()

        while penetrating:
            # Randomise angles of arm joints.
            indices = [self.model.joint_names.index(six.b(name)) for name in self.arm_joints]
            is_limited = self.model.jnt_limited[indices].astype(np.bool)
            joint_range = self.model.jnt_range[indices]
            lower_range = joint_range[:, 0]
            upper_range = joint_range[:, 1]
            lower_limits = np.where(is_limited.flatten(), lower_range, -np.pi)
            upper_limits = np.where(is_limited.flatten(), upper_range, np.pi)
            angles = np.random.uniform(lower_limits, upper_limits)
            qpos[indices] = angles

            # Symmetrize hand.
            finger_idx = self.model.joint_names.index(six.b('finger'))
            thumb_idx = self.model.joint_names.index(six.b('thumb'))
            qpos[finger_idx] = qpos[thumb_idx]

            # Fixed target location ; randomized target in goal wrapper
            # target_x = 0.
            # target_z = .2
            # target_angle = 0.
            # target_idx = self.model.body_names.index(six.b(self.target))
            # body_pos[target_idx, [0, 2]] = target_x, target_z
            # body_quat[target_idx, [0, 2]] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]

            self.model.body_pos = body_pos
            self.model.body_quat = body_quat
            self.set_state(qpos, qvel)

            # Check for collisions.
            # self.render(mode='human')
            penetrating = self.model.data.ncon > 0

        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # while True:
        #     self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #     if np.linalg.norm(self.goal) < 0.2:
        #         break
        # qpos[-2:] = self.goal
        # qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # qvel[-2:] = 0
        # self.set_state(qpos, qvel)
        return self._get_obs()

    def touch(self):
        return np.log1p(self.model.data.sensordata)

    def proprioception(self):
        """Returns the arm state, with unbounded angles as sine/cosine."""
        arm = []
        for joint in self.arm_joints:
            idx = self.model.joint_names.index(six.b(joint))
            joint_value = self.model.data.qpos[idx]
            if not self.model.jnt_limited[idx]:
                arm += [np.sin(joint_value), np.cos(joint_value)]
            else:
                arm.append(joint_value)
            # Slightly changed order of attributes
            arm.append(self.model.data.qvel[idx])
        return np.hstack(arm)

    def _get_obs(self):
        """
        position:
        sin(armroot), cos(armroot), arm_shoulder, arm_elbow, arm_wrist, thumb, thumb_tip, finger, finger_tip
        ball_x
        ball_z
        sin(ball_y)
        cos(ball_y)

        touch:
        palm, finger, thumb, fingertip, thumbtip

        full size : 33
        """
        # Sensors
        proprioception = self.proprioception() # size 9+8 = 17
        touch = self.touch() # size 5 (5 sensors)
        observations = [proprioception, touch]
        # TODO: make sure site_xpos is sufficient to describe the state. site_xmat ?

        # Objects sites for reward computation
        if self.sites:
            site_idx = [self.model.site_names.index(six.b(site)) for site in self.sites]
            site_positions = [self.model.data.site_xpos[idx][[0, 2]] for idx in site_idx]
            sites = np.hstack(site_positions) # size 2*n_sites
            observations.append(sites)

        observation_arrays = [observation.ravel() for observation in observations]
        obs = np.concatenate(observation_arrays)
        return obs