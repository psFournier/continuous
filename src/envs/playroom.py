import numpy as np
from gym import Env

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

class Actions:
    NOOP = 0
    UP=1
    DOWN=2
    RIGHT=3
    LEFT=4
    TOUCH = 5
    TAKE = 6
    # TOUCHUP = 7
    # TOUCHDOWN = 8
    # TOUCHLEFT = 9
    # TOUCHRIGHT = 10


    # PUT = 10

class Obj():
    def __init__(self, env, x, y, name, prop, dep=None):
        self.x = x
        self.y = y
        self.s = 0
        self.in_hand = 0
        self.env = env
        self.name = name
        self.prop = prop
        self.dep = dep
        self.env.objects.append(self)

    def act(self, a):
        pass

    @property
    def state(self):
        return self.s

    @property
    def high(self):
        return [1]

    @property
    def low(self):
        return [0]

    @property
    def init(self):
        return [0]

class Light(Obj):

    def __init__(self, env, x, y, name, prop):
        super(Light, self).__init__(env, x, y, name, prop)

    def act(self, a):
        if a==Actions.TOUCH and self.state == 0:
                self.s = np.random.choice([0, 1], p=self.prop)

class Key(Obj):

    def __init__(self, env, x, y, name, prop, dep):
        super(Key, self).__init__(env, x, y, name, prop, dep)

    def act(self, a):
        cond = all([d.state == 1 for d in self.dep])
        if cond and self.state == 0 and a == Actions.TAKE:
            self.in_hand = np.random.choice([0, 1], p=self.prop)

    @property
    def state(self):
        return self.in_hand

class Chest(Obj):

    def __init__(self, env, x, y, name, prop, dep):
        super(Chest, self).__init__(env, x, y, name, prop, dep)

    def act(self, a):
        # cond = all([d.state == 1 for d in self.dep])
        # if cond and a == Actions.TOUCH and self.state == 0:
        #     self.s = np.random.choice([0, 1], p=self.prop)

        if a == Actions.TOUCH:
            for d in self.dep:
                if d.state == 1:
                    self.s = np.random.choice([self.s, self.s + 1], p=self.prop)
            # if a == Actions.TOUCHDOWN:
            #     if self.s == 0:
            #         self.s = np.random.choice([0, 1], p=self.prop)
            # elif a == Actions.TOUCHUP:
            #     if self.s == 1:
            #         self.s = np.random.choice([1, 2], p=self.prop)
            # elif a == Actions.TOUCHLEFT:
            #     if self.s == 0:
            #         self.s = np.random.choice([0, 3], p=self.prop)
            # elif a == Actions.TOUCHRIGHT:
            #     if self.s == 3:
            #         self.s = np.random.choice([3, 4], p=self.prop)

    @property
    def high(self):
        return [len(self.dep)]

class Playroom(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        self.initialize()
        self.maxR = self.desc.shape[0] - 2
        self.maxC = (self.desc.shape[1] - 1) // 2 - 1

    def initialize(self):
        self.x = 0
        self.y = 0
        self.objects = []
        self.light = Light(self, 0, 3, 'light', [0.5, 0.5])
        self.key1 = Key(self, 3, 0, 'key1', [0.5, 0.5], dep=[self.light])
        self.key2 = Key(self, 7, 0, 'key2', [0.5, 0.5], dep=[self.light])
        self.key3 = Key(self, 3, 4, 'key3', [0.5, 0.5], dep=[self.light])
        self.key4 = Key(self, 7, 4, 'key4', [0.5, 0.5], dep=[self.light])
        self.chest1 = Chest(self, 2, 2, 'chest1', [0.5, 0.5], dep=[self.light, self.key1])
        self.chest2 = Chest(self, 2, 6, 'chest2', [0.5, 0.5], dep=[self.light, self.key1, self.key2])
        self.chest3 = Chest(self, 6, 2, 'chest3', [0.5, 0.5], dep=[self.light, self.key1, self.key2, self.key3])
        self.chest4 = Chest(self, 6, 6, 'chest4', [0.5, 0.5], dep=[self.light, self.key1, self.key2, self.key3, self.key4])
        self.lastaction = None

    def step(self, a):

        if a==Actions.UP and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            self.x += 1
            for h in self.get_held():
                self.objects[h].x = self.x

        elif a==Actions.DOWN and self.desc[self.x, 1 + 2 * self.y] == b" ":
            self.x -= 1
            for h in self.get_held():
                self.objects[h].x = self.x

        elif a==Actions.LEFT and self.desc[1 + self.x, 2 * self.y] == b" ":
            self.y -= 1
            for h in self.get_held():
                self.objects[h].y = self.y

        elif a==Actions.RIGHT and self.desc[1 + self.x, 2 * self.y + 2] == b" ":
            self.y += 1
            for h in self.get_held():
                self.objects[h].y = self.y

        elif a==Actions.TAKE:
            o = self.get_underagent()
            if o >= 0:
                self.objects[o].act(a)

        # elif a==Actions.PUT:
        #     h = self.get_held()
        #     o = self.get_underagent()
        #     if o == -1 and h >= 0:
        #         self.objects[h].act(a)

        elif a==Actions.TOUCH:
            o = self.get_underagent()
            if o >= 0:
                self.objects[o].act(a)
                
        # elif a==Actions.TOUCHUP:
        #     o = self.get_underagent(y = 1)
        #     if o >= 0:
        #         self.objects[o].act(a)
        #
        # elif a==Actions.TOUCHDOWN:
        #     o = self.get_underagent(y = -1)
        #     if o >= 0:
        #         self.objects[o].act(a)
        #
        # elif a==Actions.TOUCHLEFT:
        #     o = self.get_underagent(x = -1)
        #     if o >= 0:
        #         self.objects[o].act(a)
        #
        # elif a==Actions.TOUCHRIGHT:
        #     o = self.get_underagent(x = 1)
        #     if o >= 0:
        #         self.objects[o].act(a)

        elif a==Actions.NOOP:
            pass
                
        self.lastaction = a

        return np.array(self.state)

    def get_underagent(self, x=0, y=0):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x + x and obj.y == self.y + y and obj.in_hand == 0:
                return i
        return -1

    def get_held(self):
        res = []
        for i, obj in enumerate(self.objects):
            if obj.in_hand:
                res.append(i)
        return res

    def reset(self):
        self.initialize()
        return np.array(self.state)

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
            res += [obj.state]
        return res

    @property
    def init(self):
        res = [0, 0]
        for obj in self.objects:
            res += obj.init
        return res

    @property
    def low(self):
        res = [0, 0]
        for obj in self.objects:
            res += obj.low
        return res

if __name__ == '__main__':
    env = Playroom()
    env.reset()
    for a in [2,2,2,2,4,3,3,3,3,0,0,0,0,9,1,1,2,2,2,6,2,2]:
        # a = np.random.randint(11)
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
