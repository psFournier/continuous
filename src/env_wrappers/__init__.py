from env_wrappers.registration import registry, register, make, spec

register(
    id='Reacher_xy-v0',
    entry_point='envs.reacher:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher_xy:Reacher_xy'
)

register(
    id='Reacher_e-v0',
    entry_point='envs.reacher:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher_e:Reacher_e'
)

register(
    id='Reacher-v0',
    entry_point='gym.envs.mujoco:ReacherEnv',
    wrapper_entry_point=None
)

register(
    id='HalfCheetah-v2',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    wrapper_entry_point=None
)

register(
    id='FetchReach_e-v1',
    entry_point='gym.envs.robotics:FetchReachEnv',
    wrapper_entry_point='env_wrappers.fetchReach_e:FetchReach_e'
)

register(
    id='Reacher_xy_sagg-v0',
    entry_point='envs:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher_xy_sagg:Reacher_xy'
)

register(
    id='Reacher_xy_sagg_plot-v0',
    entry_point='envs:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher_xy_sagg_plot:Reacher_xy'
)