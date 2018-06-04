from ddpg.env_wrappers.registration import registry, register, make, spec

register(
    id='Reacher_xy-v0',
    entry_point='gym.envs.mujoco:ReacherEnv',
    wrapper_entry_point='ddpg.env_wrappers.reacher_xy:Reacher_xy'
)

register(
    id='Reacher_e-v0',
    entry_point='gym.envs.mujoco:ReacherEnv',
    wrapper_entry_point='ddpg.env_wrappers.reacher_e:Reacher_e'
)