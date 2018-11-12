import numpy as np
from gym import Env
from random import randint

MAP = [
    " _ _ _ _ _ _ _ _ ",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|_ _ _ _ _ _ _ _|",
]

class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    TOUCH = 4
    TAKE = 5

class Obj():
    def __init__(self, env, name, pos, prop, dep=None):
        self.env = env
        self.name = name
        self.x, self.y = pos
        self.prop = prop
        self.dep = dep
        self.env.objects.append(self)
        # self.g = self.gencoordinates()
        self.init()

    # def gencoordinates(self):
    #
    #     while True:
    #         x, y = 0, 0
    #         while (self.ix+x, self.iy+y) in self.env.seenpos:
    #             x, y = 0, 0
    #         self.env.seenpos.add((self.ix+x, self.iy+y))
    #         yield (self.ix+x, self.iy+y)

    def init(self):
        # self.x, self.y = next(self.g)
        self.s = 0

    def touch(self):
        pass

    @property
    def state(self):
        return [self.x, self.y, self.s]

    @property
    def high(self):
        return [self.env.maxR, self.env.maxC, 1]

    @property
    def low(self):
        return [0, 0, 0]

    @property
    def takeable(self):
        return True

class Light(Obj):
    def __init__(self, env, name, pos, prop):
        super(Light, self).__init__(env, name, pos, prop)

    def touch(self):
        if self.s == 0:
            self.s = np.random.choice([0, 1], p=self.prop)

    # def init(self):
    #     self.x, self.y = next(self.g)
    #     self.s = np.random.choice([0, 1], p=[0.9, 0.1])

class Key(Obj):
    def __init__(self, env, name, pos, prop, dep):
        super(Key, self).__init__(env, name, pos, prop, dep)

    def touch(self):
        if self.s == 0 and all([o.s == s for o, s in self.dep]):
            self.s = np.random.choice([0, 1], p=self.prop)

    # def init(self):
    #     self.x, self.y = next(self.g)
    #     n = 0
    #     p = [1]
    #     for o, s in self.dep:
    #         if o.s == s:
    #             n += 1
    #             p[0] -= 0.1
    #             p.append(0.1)
    #     self.s = np.random.choice(range(n+1), p=p)

class Chest(Obj):
    def __init__(self, env, name, pos, prop, dep):
        super(Chest, self).__init__(env, name, pos, prop, dep)

    def touch(self):
        if self.s == 0 and all([o.s == s for o, s in self.dep]):
            self.s = np.random.choice([0, 1], p=self.prop)

    # def init(self):
    #     self.x, self.y = next(self.g)
    #     n = 0
    #     p = [1]
    #     for o, s in self.dep[:-1]:
    #         if o.s == s:
    #             n += 1
    #             p[0] -= 0.1
    #             p.append(0.1)
    #     self.s = np.random.choice(range(n+1), p=p)

    # @property
    # def high(self):
    #     return [self.env.maxR, self.env.maxC, len(self.dep)]

class Playroom(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.maxR = self.desc.shape[0] - 2
        self.maxC = (self.desc.shape[1] - 1) // 2 - 1
        self.initialize()

    def initialize(self, random=True):
        self.seenpos = set()
        self.x, self.y = randint(0, self.maxR), randint(0, self.maxC)
        self.seenpos.add((self.x, self.y))
        self.obj = 0
        self.objects = []
        self.light = Light(self,
                           name='light',
                           pos=(1,1),
                           prop=[0, 1])
        self.key1 = Key(self,
                        name='key1',
                        pos=(4,4),
                        prop=[0, 1],
                        dep=[(self.light, 1)])
        self.chest1 = Chest(self,
                            name='chest1',
                            pos=(6,6),
                            prop=[0, 1],
                            dep=[(self.light, 1), (self.key1, 1)])
        if not random:
            self.light.s = 0
            self.key1.s = 0
            self.chest1.s = 0

        self.initstate = self.state.copy()
        self.lastaction = None

    def step(self, a):
        if a == Actions.UP:
            if self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
                self.x += 1
        elif a == Actions.DOWN:
            if self.desc[self.x, 1 + 2 * self.y] == b" ":
                self.x -= 1
        elif a == Actions.LEFT:
            if self.desc[1 + self.x, 2 * self.y] == b" ":
                self.y -= 1
        elif a == Actions.RIGHT:
            if self.desc[1 + self.x, 2 * self.y + 2] == b" ":
                self.y += 1
        elif a == Actions.TAKE:
            obj = self.underagent()
            if self.obj == 0 and obj != 0:
                self.obj = obj
            elif obj == 0:
                self.obj = 0
        elif a == Actions.TOUCH:
            if self.obj != 0:
                self.objects[self.obj - 1].touch()
            else:
                obj = self.underagent()
                if obj != 0:
                    self.objects[obj - 1].touch()

        if self.obj != 0:
            self.objects[self.obj - 1].x, self.objects[self.obj - 1].y = self.x, self.y

        self.lastaction = a
        return np.array(self.state),

    def underagent(self):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x and obj.y == self.y and self.obj != i+1:
                return i+1
        return 0

    def reset(self, random=True):
        self.initialize(random)
        return np.array(self.state)

    def go(self, x , y):
        dx = x - self.x
        dy = y - self.y
        possible_act = []
        if dx > 0:
            possible_act.append(Actions.UP)
        elif dx < 0:
            possible_act.append(Actions.DOWN)
        elif dy > 0:
            possible_act.append(Actions.RIGHT)
        elif dy < 0:
            possible_act.append(Actions.LEFT)
        if possible_act:
            return np.random.choice(possible_act)
        else:
            return None

    def take(self, i):
        obj = self.objects[i]
        a = self.go(obj.x, obj.y)
        if a is None:
            return Actions.TAKE, False
        else:
            return a, False

    def touch(self, o):
        a = self.go(o.x, o.y)
        if a is None:
            return Actions.TOUCH, False
        else:
            return a, False

    @property
    def high(self):
        res = [self.maxR, self.maxC, len(self.objects)]
        for obj in self.objects:
            res += obj.high
        return res

    @property
    def state(self):
        res = [self.x, self.y, self.obj]
        for obj in self.objects:
            res += obj.state
        return res

    @property
    def low(self):
        res = [0, 0, 0]
        for obj in self.objects:
            res += obj.low
        return res

if __name__ == '__main__':
    env = Playroom()
    env.reset()
    print(env.state)
    while True:
        a, done = env.opt_action(6, 4)
        if not done:
            env.step(a)
        else:
            break
        print(a, env.state)


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
