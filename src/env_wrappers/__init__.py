from .base import CPBased, Base
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
register(
    id='ReacherShaped-v0',
    entry_point='envs.reacher:ReacherBaseEnv',
    wrapper_entry_point='env_wrappers.reacher:ReacherWrapShaped'
)

register(
    id='Reacher-v0',
    entry_point='envs.reacher:ReacherBaseEnv',
    wrapper_entry_point='env_wrappers.reacher:ReacherWrap'
)

register(
    id='ReacherDeceptShaped-v0',
    entry_point='envs.reacher:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher:ReacherWrapShaped'
)

register(
    id='ReacherDecept-v0',
    entry_point='envs.reacher:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher:ReacherWrap'
)

register(
    id='ReacherDeceptEpsCP-v0',
    entry_point='envs.reacher:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher_e:Reacher_e'
)

register(
    id='ReacherDeceptEpsRnd-v0',
    entry_point='envs.reacher:ReacherEnv',
    wrapper_entry_point='env_wrappers.reacher_e2:Reacher_e2'
)
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
    id='Taxi1G-v0',
    entry_point='envs:Taxi1',
    wrapper_entry_point='env_wrappers.taxi1G:Taxi1G'
)

register(
    id='Taxi2GM-v0',
    entry_point='envs:Taxi2',
    wrapper_entry_point='env_wrappers.taxi2GM:Taxi2GM'
)

register(
    id='Playroom-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroom:Playroom'
)

register(
    id='PlayroomG-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroomG:PlayroomG'
)

register(
    id='PlayroomGM-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroomGM:PlayroomGM'
)

register(
    id='PlayroomGM2-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroomGM2:PlayroomGM2'
)

register(
    id='PlayroomGO-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroomGO:PlayroomGO'
)

register(
    id='PlayroomGF-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroomGF:PlayroomGF'
)

register(
    id='Labyrinth-v0',
    entry_point='envs:Labyrinth',
    wrapper_entry_point='env_wrappers.labyrinth:Labyrinth'
)

register(
    id='Labyrinth2-v0',
    entry_point='envs:Labyrinth',
    wrapper_entry_point='env_wrappers.labyrinth2:Labyrinth2'
)

register(
    id='LabyrinthG-v0',
    entry_point='envs:Labyrinth',
    wrapper_entry_point='env_wrappers.labyrinthG:LabyrinthG'
)