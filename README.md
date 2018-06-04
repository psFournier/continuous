# continuous

This repo contains an implementaiton of the Deep Deterministic Policy Gradients algorithm and Hindsight Experience Replay, along with two additional ingredients:
  - environment wrappers classes inherited from the OpenAI gym and augmented to implement Universal Value Function Approximators (UVFA)
  - a principled goal selection mechanism based on competence progress measured in different regions of the environment
  
Environments supported are those registered in src/ddpg/env_wrappers/__init__.py. Following the OpenAI gym Registry mechanism, each entry specifies the env id as well as the locations of its class and wrapper.

The environment wrappers allow to augment the state with some broad notion of goal, and redefine the step and reset methods to take this addition into account. For now, a goal can be defined as:
  - x,y coordinates of a possible target in the observation space (Reacher_xy)
  - a precision epsilon with which the agent should reach random goals picked in the observation space (Reacher_e, FetchReach_e)
  
To simply use the original environment with no wrappers, the wrapper_entry_point is simplu set to None (Reacher)
