from .base import CPBased, Base
from env_wrappers.registration import registry, register, make

register(
    id='PlayroomGM-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroomGM:PlayroomGM'
)