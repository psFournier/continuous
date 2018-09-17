import numpy as np
from gym import Env

'''self.light = Light(self, 2, 3)
        self.key1 = Key1(self, 0, 3)
        self.chest1 = Chest1(self, 3, 2)
        self.chest2 = Chest2(self, 5, 1)
        self.chest3 = Chest3(self, 4, 6)'''

MAP = [
    "+-------------+",
    "|    _ _|     |",
    "|       :     |",
    "|_ _ _  |     |",
    "|    _| |     |",
    "|    _ _|   | |",
    "|       |   | |",
    "|       |   | |",
    "+-------------+",
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
    def __init__(self, env, x, y):
        self.x = x
        self.y = y
        self.s = 0
        self.in_hand = 0
        self.env = env

    def act(self, a):
        pass

    @property
    def light(self):
        return self.env.light.s == 1

class Light(Obj):
    def __init__(self, env, x, y):
        super(Light, self).__init__(env, x, y)
        self.name = 'light'

    def act(self, a):
        if a==Actions.TOUCH:
            self.s = 1 - self.s

    @property
    def smax(self):
        return 1

class Key1(Obj):
    def __init__(self, env, x, y):
        super(Key1, self).__init__(env, x, y)
        self.name = 'key1'

    def act(self, a):
        if a == Actions.TAKE:
            if self.light:
                self.in_hand = 1

    @property
    def smax(self):
        return 0

class Chest1(Obj):
    def __init__(self, env, x, y):
        super(Chest1, self).__init__(env, x, y)
        self.name = 'chest1'

    def act(self, a):
        if a == Actions.TOUCH:
            if self.light:
                self.s = np.random.choice(range(4), p=[0.2, 0.4, 0.2, 0.2])

    @property
    def smax(self):
        return 3

class Chest2(Obj):
    def __init__(self, env, x, y):
        super(Chest2, self).__init__(env, x, y)
        self.name = 'chest2'

    def act(self, a):
        if self.light:
            if a == Actions.TOUCHDOWN:
                if self.s == 0:
                    self.s = 1
            elif a == Actions.TOUCHUP:
                if self.s == 1:
                    self.s = 2
            elif a == Actions.TOUCHLEFT:
                if self.s == 2:
                    self.s = 3
            elif a == Actions.TOUCHRIGHT:
                if self.s == 3:
                    self.s = 4
            elif a == Actions.TOUCH:
                if self.s > 0:
                    self.s = 0

    @property
    def smax(self):
        return 4

class Chest3(Obj):
    def __init__(self, env, x, y):
        super(Chest3, self).__init__(env, x, y)
        self.name = 'chest3'

    def act(self, a):
        if a == Actions.TOUCH:
            if self.light:
                self.s = 1

    @property
    def smax(self):
        return 1

class Playroom(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        self.maxR = 6
        self.maxC = 6
        self.init()

    def init(self):
        self.x = 0
        self.y = 0
        self.light = Light(self, 2, 3)
        self.key1 = Key1(self, 0, 3)
        self.chest1 = Chest1(self, 3, 2)
        self.chest2 = Chest2(self, 5, 2)
        self.chest3 = Chest3(self, 4, 6)
        self.objects = [self.key1, self.chest1, self.chest2, self.chest3, self.light]
        self.lastaction = None

    def step(self, a):

        # print(self.get_state(), a)

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
        self.init()
        return np.array(self.state)

    @property
    def state_high(self):
        smax = [self.maxR , self.maxC]
        for o in self.objects:
            smax += [self.maxR, self.maxC, o.smax, 1]
        return smax

    @property
    def state(self):
        res = [self.x, self.y]
        for obj in self.objects:
            res += [obj.x, obj.y, obj.s, obj.in_hand]
        return res

    @property
    def state_init(self):
        return [0,0,0,3,0,0,3,2,0,0,5,2,0,0,4,6,0,0,2,3,0,0]

    @property
    def state_low(self):
        return [0] * (2 + len(self.objects) * 4)

if __name__ == '__main__':
    env = Playroom()
    env.reset()
    for a in [0, 0, 2, 2, 2, 4, 0, 0, 0,3,3,3,1,2,2,4, 4, 4, 4, 4]:
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
