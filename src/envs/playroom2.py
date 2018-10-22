import numpy as np
from gym import Env
from random import randint

'''self.light = Light(self, 2, 3)
        self.key1 = Key1(self, 0, 3)
        self.chest1 = Chest1(self, 3, 2)
        self.chest2 = Chest2(self, 5, 1)
        self.chest3 = Chest3(self, 4, 6)'''

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


# class Moves:
#     NOOP = 0
#     UP = 1
#     DOWN = 2
#     LEFT = 3
#     RIGHT = 4
#
# class Actions:
#     NOOP = 0
#     TOUCH = 1
#     TAKE = 2
#     DROP = 3

class Actions:
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    TOUCH = 5

# class Moves:
#     UP = 0
#     DOWN = 1
#     LEFT = 2
#     RIGHT =3
    # TOUCHUP = 7
    # TOUCHDOWN = 8
    # TOUCHLEFT = 9
    # TOUCHRIGHT = 10
    # PUT = 10

def gencoordinates(m, n):
    seen = set()

    x, y = randint(m, n), randint(m, n)

    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(m, n), randint(m, n)
        while (x, y) in seen:
            x, y = randint(m, n), randint(m, n)

class Obj():
    def __init__(self, env, pos, name, prop, dep=None):
        self.x, self.y = pos
        self.s = 0
        # self.inhand = 0
        self.env = env
        self.name = name
        self.prop = prop
        self.dep = dep
        self.env.objects.append(self)

    def act(self, a):
        pass

    @property
    def state(self):
        return [self.x, self.y, self.s]
        # return [self.x, self.y, self.s, self.inhand]

    @property
    def high(self):
        return [self.env.maxR, self.env.maxC, 1]
        # return [self.env.maxR, self.env.maxC, 1, 1]

    @property
    def low(self):
        return [0, 0, 0]
        # return [0, 0, 0, 0]

class Light(Obj):
    def __init__(self, env, pos, name, prop):
        super(Light, self).__init__(env, pos, name, prop)

    def act(self, a):
        if a == Actions.TOUCH and self.s == 0:
            self.s = np.random.choice([0, 1], p=self.prop)

    # @property
    # def state(self):
    #     return [self.x, self.y, self.s]
    #
    # @property
    # def high(self):
    #     return [self.env.maxR, self.env.maxC, 1]
    #
    # @property
    # def low(self):
    #     return [0, 0, 0]

class Key(Obj):
    def __init__(self, env, pos, name, prop, dep):
        super(Key, self).__init__(env, pos, name, prop, dep)

    def act(self, a):
        # if self.env.light.s == 1 and a == Actions.TAKE:
        #     self.inhand = np.random.choice([0, 1], p=self.prop)
        if self.env.light.s == 1 and a == Actions.TOUCH and self.s == 0:
            self.s = np.random.choice([0, 1], p=self.prop)

    # @property
    # def state(self):
    #     return [self.x, self.y, self.inhand]
    #
    # @property
    # def high(self):
    #     return [self.env.maxR, self.env.maxC, 1]
    #
    # @property
    # def low(self):
    #     return [0, 0, 0]

class Chest(Obj):
    def __init__(self, env, pos, name, prop, dep):
        super(Chest, self).__init__(env, pos, name, prop, dep)

    def act(self, a):
        if a == Actions.TOUCH and self.s < len(self.dep):
            obj = self.dep[self.s]
            if obj.s == 1:
            # if (isinstance(obj, Light) and obj.s == 1) or (isinstance(obj, Key) and obj.inhand == 1):
                new_s = np.random.choice([self.s, self.s + 1], p=self.prop)
                if new_s != self.s:
                    # obj.s = 0
                    self.s = new_s
        # if self.env.light.s == 1 and a == Actions.TAKE:
        #     self.inhand = np.random.choice([0, 1], p=self.prop)

    # @property
    # def state(self):
    #     return [self.x, self.y, self.s]

    @property
    def high(self):
        return [self.env.maxR, self.env.maxC, len(self.dep)]

    # @property
    # def low(self):
    #     return [0, 0, 0]


class Playroom2(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.maxR = self.desc.shape[0] - 2
        self.maxC = (self.desc.shape[1] - 1) // 2 - 1
        self.initialize()

    def initialize(self):
        self.g = gencoordinates(0, self.maxR)
        self.x, self.y = next(self.g)
        self.objects = []
        prop = [0, 1]
        self.light = Light(self, next(self.g), 'light', prop)
        self.key1 = Key(self, next(self.g), 'key1', prop, dep=[self.light])
        # self.key2 = Key(self, next(self.g), 'key2', prop, dep=[self.light])
        # self.key3 = Key(self, next(self.g), 'key3', prop, dep=[self.light])
        # self.key4 = Key(self, next(self.g), 'key4', prop, dep=[self.light])
        # self.chest0 = Chest(self, next(self.g), 'chest0', prop, dep=[self.light])
        self.chest1 = Chest(self, next(self.g), 'chest1', prop, dep=[self.light, self.key1])
        # self.chest2 = Chest(self, next(self.g), 'chest2', prop, dep=[self.light, self.key1, self.key2])
        # self.chest3 = Chest(self, next(self.g), 'chest3', prop, dep=[self.light, self.key1, self.key2, self.key3])
        # self.chest4 = Chest(self, next(self.g), 'chest4', prop,
        #                     dep=[self.light, self.key1, self.key2, self.key3, self.key4])
        self.initstate = self.state.copy()
        self.lastaction = None

    def act(self, a):
        # objinhand = self.inhand()
        objunder = self.underagent()

        # if objinhand is not None:
        #     objinhand.x = self.x
        #     objinhand.y = self.y

        # if objinhand is not None and objunder is None and a == Actions.DROP:
        #     objinhand.inhand = 0
        #
        # if objinhand is None and objunder is not None and a == Actions.TAKE:
        #     objunder.act(a)

        if objunder is not None and a == Actions.TOUCH:
            objunder.act(a)

    def step(self, a):
        if a == Actions.UP and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            self.x += 1
        elif a == Actions.DOWN and self.desc[self.x, 1 + 2 * self.y] == b" ":
            self.x -= 1
        elif a == Actions.LEFT and self.desc[1 + self.x, 2 * self.y] == b" ":
            self.y -= 1
        elif a == Actions.RIGHT and self.desc[1 + self.x, 2 * self.y + 2] == b" ":
            self.y += 1
        self.act(a)
        self.lastaction = a
        return np.array(self.state),

    def underagent(self):
        for obj in self.objects:
            if obj.x == self.x and obj.y == self.y:
                return obj
        return None

    # def inhand(self):
    #     for obj in self.objects:
    #         if obj.inhand == 1:
    #             return obj
    #     return None

    def reset(self):
        self.initialize()
        return np.array(self.state)

    def go(self, x , y):
        dx = x - self.x
        dy = y - self.y
        if dx > 0:
            a = Actions.UP
        elif dx < 0:
            a = Actions.DOWN
        elif dy > 0:
            a = Actions.RIGHT
        elif dy < 0:
            a = Actions.LEFT
        else:
            a = None
        return a

    def opt_action(self, obj, goal):
        if obj is None:
            a = self.go(goal[0], goal[1])
            if a is not None:
                return a, False
            else:
                return Actions.NOOP, True
        else:
            if obj.dep:
                for i, o in enumerate(obj.dep[:goal]):
                    if o.s < 1:
                        a = self.go(o.x, o.y)
                        if a is None:
                            return Actions.TOUCH, False
                        else:
                            return a, False
            if obj.s < goal:
                a = self.go(obj.x, obj.y)
                if a is None:
                    return Actions.TOUCH, False
                else:
                    return a, False
            else:
                return Actions.NOOP, True

    @property
    def high(self):
        res = [self.maxR, self.maxC]
        for obj in self.objects:
            res += obj.high
        return res

    @property
    def state(self):
        res = [self.x, self.y]
        for obj in self.objects:
            res += obj.state
        return res

    @property
    def low(self):
        res = [0, 0]
        for obj in self.objects:
            res += obj.low
        return res

if __name__ == '__main__':
    env = Playroom2()
    env.reset()
    while True:
        a, done = env.opt_action(env.chest1, 2)
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
