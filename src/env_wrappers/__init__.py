from env_wrappers.registration import registry, register, make, spec

# register(
#     id='Reacher_xy-v0',
#     entry_point='envs.reacher:ReacherEnv',
#     wrapper_entry_point='env_wrappers.reacher_xy:Reacher_xy'
# )
#
# register(
#     id='Reacher_e-v0',
#     entry_point='envs.reacher:ReacherEnv',
#     wrapper_entry_point='env_wrappers.reacher_e:Reacher_e'
# )
#
# register(
#     id='Reacher-v0',
#     entry_point='envs.reacher:ReacherEnv',
#     wrapper_entry_point=None
# )
#
# register(
#     id='HalfCheetah-v2',
#     entry_point='gym.envs.mujoco:HalfCheetahEnv',
#     wrapper_entry_point=None
# )
#
# register(
#     id='Ant-v2',
#     entry_point='gym.envs.mujoco:AntEnv',
#     wrapper_entry_point=None
# )
#
# register(
#     id='FetchReach_e-v1',
#     entry_point='gym.envs.robotics:FetchReachEnv',
#     wrapper_entry_point='env_wrappers.fetchReach_e:FetchReach_e'
# )
#
# register(
#     id='Reacher_xy_sagg-v0',
#     entry_point='envs:ReacherEnv',
#     wrapper_entry_point='env_wrappers.reacher_xy_sagg:Reacher_xy'
# )
#
# register(
#     id='Reacher_xy_sagg_plot-v0',
#     entry_point='envs:ReacherEnv',
#     wrapper_entry_point='env_wrappers.reacher_xy_sagg_plot:Reacher_xy'
# )

# register(
#     id='Taxi-v0',
#     entry_point='envs:TaxiEnv',
#     wrapper_entry_point='env_wrappers.taxi_gamma:Taxi_gamma'
# )
#
# register(
#     id='Taxi-v1',
#     entry_point='envs:TaxiEnv',
#     wrapper_entry_point='env_wrappers.taxi_goal:TaxiGoal'
# )

# register(
#     id='Taxi-v0',
#     entry_point='envs:TaxiEnv',
#     wrapper_entry_point='env_wrappers.taxi:Taxi'
# )
#


# register(
#     id='TaxiTutor-v0',
#     entry_point='envs:TaxiEnv',
#     wrapper_entry_point='env_wrappers.taxi_goal_tutor:TaxiGoalTutor'
# )

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    wrapper_entry_point=None
)

register(
    id='TaxiGoal-v0',
    entry_point='envs:TaxiEnv',
    wrapper_entry_point='env_wrappers.taxiGoal:TaxiGoal'
)

register(
    id='TaxiGoalMask-v0',
    entry_point='envs:Taxi2Env',
    wrapper_entry_point='env_wrappers.taxiGoalMask:TaxiGoalMask'
)

register(
    id='TaxiGoal2-v0',
    entry_point='envs:Taxi2Env',
    wrapper_entry_point='env_wrappers.taxiGoal2:TaxiGoal2'
)

register(
    id='PlayroomMask-v0',
    entry_point='envs:PlayroomEnv',
    wrapper_entry_point='env_wrappers.playroomMask:PlayroomMask'
)