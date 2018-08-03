import numpy as np
from gym import Env

MAP = [
    "+-------------+",
    "| : : : : : : |",
    "| : : : : : : |",
    "| : : : : : : |",
    "| : : : : : : |",
    "| : : : : : : |",
    "| : : : : : : |",
    "| : : : : : : |",
    "+-------------+",
]
class Actions:
    UP=0
    DOWN=1
    RIGHT=2
    LEFT=3
    LOOK_LIGHT = 4
    LOOK_BELL = 5
    LOOK_MUSIC = 6
    LOOK_CUBE = 7
    LOOK_BOX = 8
    LOOK_TOY = 9
    TOUCH = 10
    TAKE = 11
    PUT = 12

class Features:
    X_HAND=0
    Y_HAND=1
    X_EYE=2
    Y_EYE=3
    POS_LIGHT = 4
    POS_BELL = 5
    POS_MUSIC = 6
    POS_CUBE = 7
    POS_BOX = 8
    POS_TOY = 9
    SOUND = 10
    LIGHT = 11
    TOY = 12
    UNDER_EYE=13

class Objects:
    LIGHT = 0
    BELL = 1
    MUSIC = 2
    CUBE = 3
    BOX = 4
    TOY = 5

class PlayroomEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        self.s = None
        self.lastaction = None
        self.maxR = 6
        self.maxC = 6
        self.nObjects = 6
        self.init_pos = {
            'light': (3,3),
            'bell':(0,4),
            'music': (5,1),
            'cube': (2,6),
            'box': (5,5),
            'toy': (2,3)
        }

    def step(self, a):

        if a==Actions.UP:
            self.s[Features.Y_HAND] = min(self.s[Features.Y_HAND] + 1, self.maxR)

        elif a==Actions.DOWN:
            self.s[Features.Y_HAND] = max(self.s[Features.Y_HAND] - 1, 0)

        elif a==Actions.LEFT:
            self.s[Features.X_HAND] = max(self.s[Features.X_HAND] - 1, 0)

        elif a==Actions.RIGHT:
            self.s[Features.X_HAND] = min(self.s[Features.X_HAND] + 1, self.maxC)

        elif a==Actions.LOOK_LIGHT and self.s[Features.POS_LIGHT]==0:
            self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['light']

        elif a==Actions.LOOK_BELL and self.s[Features.POS_BELL]==0:
            self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['bell']

        elif a==Actions.LOOK_MUSIC and self.s[Features.POS_MUSIC]==0:
            self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['music']

        elif a==Actions.LOOK_CUBE and self.s[Features.POS_CUBE]==0:
            self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['cube']

        elif a==Actions.LOOK_BOX and self.s[Features.POS_BOX]==0:
            self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['box']

        elif a==Actions.LOOK_TOY and self.s[Features.POS_TOY]==0:
            self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['toy']

        elif a==Actions.TAKE and self.hand_on_eye and self.s[Features.UNDER_EYE]!=Objects.BOX:
            self.s[4 + self.s[Features.UNDER_EYE]] = 1

        elif a==Actions.TOUCH and self.hand_on_eye and self.s[Features.UNDER_EYE]==Objects.MUSIC:
            self.s[Features.SOUND] = 1

        elif a==Actions.TOUCH and self.hand_on_eye and self.s[Features.UNDER_EYE]==Objects.LIGHT:
            self.s[Features.LIGHT] = 1

        elif a==Actions.TOUCH and self.hand_on_eye and self.s[Features.UNDER_EYE]==Objects.TOY:
            self.s[Features.TOY] = 1

        elif a==Actions.PUT and self.hand_on_eye and self.s[Features.UNDER_EYE]==Objects.BOX:
            for o in range(self.nObjects):
                if self.s[4 + o] == 1:
                    self.s[4 + o] = 2

        self.lastaction = a
        return (self.s, 0, False, {"prob" : 1})

    @property
    def hand_on_eye(self):
        return (self.s[Features.X_EYE], self.s[Features.Y_EYE]) == (self.s[Features.X_HAND], self.s[Features.Y_HAND])

    def reset(self):
        self.s = np.zeros(shape=(13,1))
        self.s[Features.X_EYE], self.s[Features.Y_EYE] = self.init_pos['light']
        self.lastaction=None
        return self.s

    # def render(self, mode='human'):
    #     outfile = StringIO() if mode == 'ansi' else sys.stdout
    #
    #     out = self.desc.copy().tolist()
    #     out = [[c.decode('utf-8') for c in line] for line in out]
    #     taxirow, taxicol, passidx = self.decode(self.s)
    #     def ul(x): return "_" if x == " " else x
    #     if passidx < 4:
    #         out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
    #         pi, pj = self.locs[passidx]
    #         out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
    #     else: # passenger in taxi
    #         out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)
    #
    #     # No need to return anything for human
    #     if mode != 'human':
    #         return outfile
