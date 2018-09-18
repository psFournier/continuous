import numpy as np
from gym import Env

'''self.light = Light(self, 2, 3)
        self.key1 = Key1(self, 0, 3)
        self.chest1 = Chest1(self, 3, 2)
        self.chest2 = Chest2(self, 5, 1)
        self.chest3 = Chest3(self, 4, 6)'''

MAP = [
    "+-------------------+",
    "|         |         |",
    "|                   |",
    "|                   |",
    "|                   |",
    "|_       _|_       _|",
    "|         |         |",
    "|                   |",
    "|                   |",
    "|                   |",
    "|         |         |",
    "+-------------------+",
]

class Actions:
    UP=0
    DOWN=1
    RIGHT=2
    LEFT=3
    TOUCH = 4
    TOUCHUP = 5
    TOUCHDOWN = 6
    TOUCHLEFT = 7
    TOUCHRIGHT = 8
    TAKE = 9
    # PUT = 10

class Obj():
    def __init__(self, env, x, y, name):
        self.x = x
        self.y = y
        self.s = 0
        self.in_hand = 0
        self.env = env
        self.name = name
        self.env.objects.append(self)

    def act(self, a):
        pass

class Light(Obj):
    def __init__(self, env, x, y, name):
        super(Light, self).__init__(env, x, y, name)

    def act(self, a):
        if a==Actions.TOUCH:
            self.s = 1 - self.s

    @property
    def high(self):
        return [1]

    @property
    def low(self):
        return [0]

    @property
    def state(self):
        return [self.s]

    @property
    def init(self):
        return [0]

class Key(Obj):
    def __init__(self, env, x, y, name, light, prop):
        super(Key, self).__init__(env, x, y, name)
        self.light = light
        self.prop = prop

    def act(self, a):
        if a == Actions.TAKE:
            if self.light.s == 1:
                self.in_hand = np.random.choice([0, 1], p=self.prop)

    @property
    def state(self):
        return [self.in_hand]

    @property
    def high(self):
        return [1]

    @property
    def low(self):
        return [0]

    @property
    def init(self):
        return [0]

class Chest(Obj):
    def __init__(self, env, x, y, name, light, key, prop):
        super(Chest, self).__init__(env, x, y, name)
        self.light = light
        self.prop = prop
        self.key = key

    def act(self, a):
        if self.light.s == 1 and self.key.in_hand == 1:
            if a == Actions.TOUCHDOWN:
                if self.s == 0:
                    self.s = np.random.choice([0, 1], p=self.prop)
            elif a == Actions.TOUCHUP:
                if self.s == 1:
                    self.s = np.random.choice([1, 2], p=self.prop)
            elif a == Actions.TOUCHLEFT:
                if self.s == 2:
                    self.s = np.random.choice([2, 3], p=self.prop)
            elif a == Actions.TOUCHRIGHT:
                if self.s == 3:
                    self.s = np.random.choice([3, 4], p=self.prop)
            elif a == Actions.TOUCH:
                if self.s > 0:
                    self.s = 0

    @property
    def high(self):
        return [4]

    @property
    def low(self):
        return [0]

    @property
    def state(self):
        return [self.s]

    @property
    def init(self):
        return [0]

class Playroom(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        self.maxR = 6
        self.maxC = 6
        self.initialize()

    def initialize(self):
        self.x = 0
        self.y = 0
        self.objects = []
        self.light1 = Light(self, 0, 4, 'light1')
        self.light2 = Light(self, 0, 9, 'light2')
        self.light3 = Light(self, 5, 4, 'light3')
        self.light4 = Light(self, 5, 9, 'light4')
        self.key1 = Key(self, 4, 0, 'key1', self.light1, [0, 1])
        self.key2 = Key(self, 4, 5, 'key2', self.light1, [0.2, 0.8])
        self.key3 = Key(self, 9, 0, 'key3', self.light1, [0.4, 0.6])
        self.key4 = Key(self, 9, 5, 'key4', self.light1, [0.6, 0.4])
        self.chest1 = Chest(self, 2, 2, 'chest1', self.light1, self.key1, [0, 1])
        self.chest2 = Chest(self, 2, 7, 'chest2', self.light2, self.key2, [0.2, 0.8])
        self.chest3 = Chest(self, 7, 2, 'chest3', self.light3, self.key3, [0.4, 0.6])
        self.chest3 = Chest(self, 7, 7, 'chest4', self.light4, self.key4, [0.6, 0.4])
        self.lastaction = None

    def step(self, a):

        if a==Actions.UP and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            self.x = min(self.x + 1, self.maxR)
            h = self.get_held()
            if h >= 0:
                self.objects[h].x = self.x

        elif a==Actions.DOWN and self.desc[self.x, 1 + 2 * self.y] == b" ":
            self.x = max(self.x - 1, 0)
            h = self.get_held()
            if h >= 0:
                self.objects[h].x = self.x

        elif a==Actions.LEFT:
            h = self.get_held()
            if self.desc[1 + self.x, 2 * self.y] == b" ":
                self.y = max(self.y - 1, 0)
                if h >= 0:
                    self.objects[h].y = self.y
            elif self.desc[1 + self.x, 2 * self.y] == b":" and h==0:
                self.y = max(self.y - 1, 0)
                self.objects[h].y = self.y

        elif a==Actions.RIGHT:
            h = self.get_held()
            if self.desc[1 + self.x, 2 * self.y + 2] == b" ":
                self.y = min(self.y + 1, self.maxC)
                if h >= 0:
                    self.objects[h].y = self.y
            elif self.desc[1 + self.x, 2 * self.y + 2] == b":" and h==0:
                self.y = min(self.y + 1, self.maxC)
                self.objects[h].y = self.y

        elif a==Actions.TAKE:
            h = self.get_held()
            o = self.get_underagent()
            if o >= 0 and h == -1:
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
                
        elif a==Actions.TOUCHUP:
            o = self.get_underagent(y = 1)
            if o >= 0:
                self.objects[o].act(a)
                
        elif a==Actions.TOUCHDOWN:
            o = self.get_underagent(y = -1)
            if o >= 0:
                self.objects[o].act(a)
                
        elif a==Actions.TOUCHLEFT:
            o = self.get_underagent(x = -1)
            if o >= 0:
                self.objects[o].act(a)
                
        elif a==Actions.TOUCHRIGHT:
            o = self.get_underagent(x = 1)
            if o >= 0:
                self.objects[o].act(a)
                
        self.lastaction = a

        return np.array(self.state)

    def get_underagent(self, x=0, y=0):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x + x and obj.y == self.y + y and obj.in_hand == 0:
                return i
        return -1

    def get_held(self):
        for i, obj in enumerate(self.objects):
            if obj.in_hand:
                return i
        return -1

    def reset(self):
        self.initialize()
        return np.array(self.state)

    @property
    def high(self):
        res = [9, 9]
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
