import numpy as np
from .playroomGM import PlayroomGM

class PlayroomGM3(PlayroomGM):
    def __init__(self, env, args):
        super(PlayroomGM3, self).__init__(env, args)

    def augment_exp(self, exp):

        obj = self.get_idx()
        mask = self.obj2mask(obj)