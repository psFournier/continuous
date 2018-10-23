import numpy as np
from .playroom2GM import Playroom2GM

class Playroom2GM0(Playroom2GM):
    def __init__(self, env, args):
        super(Playroom2GM0, self).__init__(env, args)

    def augment_demo(self, demo):
        augmented_demo = self.process_trajectory(demo, [], [], [], [])
        return augmented_demo