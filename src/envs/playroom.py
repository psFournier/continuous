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
    TAKE = 4
    PUT = 5
    TOUCH = 6

class Obj():
    def __init__(self):
        self.initx = 0
        self.inity = 0
        self.s = 0
        self.in_hand = 0

    def act(self, a):
        pass

class Light(Obj):
    def __init__(self, initx=0, inity=0):
        super(Light, self).__init__()
        self.initx = initx
        self.inity = inity
        self.x = self.initx
        self.y = self.inity

    def act(self, a):
        if a==Actions.TOUCH:
            self.s = 1 - self.s

class Toy1(Obj):
    def __init__(self, light, initx=0, inity=0):
        super(Toy1, self).__init__()
        self.initx = initx
        self.inity = inity
        self.x = self.initx
        self.y = self.inity
        self.light = light

    def act(self, a):
        if a == Actions.TOUCH and self.light.s == 1:
            self.s = 1 - self.s
        elif a == Actions.TAKE:
            self.in_hand = 1
        elif a == Actions.PUT:
            self.in_hand = 0

class PlayroomEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')

        self.x = 0
        self.y = 0
        light = Light(initx=3, inity=4)
        obj1 = Toy1(light, initx=2, inity=2)

        self.objects = [light, obj1]
        self.lastaction = None
        self.maxR = 6
        self.maxC = 6

    def step(self, a):

        if a==Actions.UP:
            self.y = min(self.y + 1, self.maxR)
            if self.held >= 0:
                self.objects[self.held].y = self.y

        elif a==Actions.DOWN:
            self.y = max(self.y - 1, 0)
            if self.held >= 0:
                self.objects[self.held].y = self.y

        elif a==Actions.LEFT:
            self.x = max(self.x - 1, 0)
            if self.held >= 0:
                self.objects[self.held].x = self.x

        elif a==Actions.RIGHT:
            self.x = min(self.x + 1, self.maxC)
            if self.held >= 0:
                self.objects[self.held].x = self.x

        elif a==Actions.TAKE and self.held == -1:
            object_idx = self.underagent
            if object_idx >= 0:
                self.objects[object_idx].act(a)

        elif a==Actions.PUT and self.held >= 0:
            object_idx = self.underagent
            if object_idx == -1:
                self.objects[self.held].act(a)

        elif a==Actions.TOUCH:
            object_idx = self.underagent
            if object_idx >= 0:
                self.objects[object_idx].act(a)
                
        self.lastaction = a
        return (self.state, 0, False, {"prob" : 1})

    @property
    def underagent(self):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x and obj.y == self.y and obj.in_hand == 0:
                return i
        return -1

    @property
    def state(self):
        res = [self.x, self.y]
        for obj in self.objects:
            res += [obj.x, obj.y, obj.s, obj.in_hand]
        return res

    @property
    def held(self):
        for i, obj in enumerate(self.objects):
            if obj.in_hand:
                return i
        return -1

    def reset(self):
        self.x = 0
        self.y = 0
        for obj in self.objects:
            obj.x = obj.initx
            obj.y = obj.inity
            obj.s = 0
            obj.in_hand = 0
        self.lastaction = None
        return self.state

if __name__ == '__main__':
    env = PlayroomEnv()
    env.reset()
    for _ in range(10000):
        a = np.random.randint(7)
        env.step(a)

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
