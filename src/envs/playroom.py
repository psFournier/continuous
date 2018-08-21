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
    TOUCHUP = 7
    TOUCHDOWN = 8
    TOUCHLEFT = 9
    TOUCHRIGHT = 10

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
        self.name = 'light'

    def act(self, a):
        if a==Actions.TOUCH:
            self.s = 1 - self.s

    @property
    def smax(self):
        return 1

class Sound(Obj):
    def __init__(self, light, initx=0, inity=0):
        super(Sound, self).__init__()
        self.initx = initx
        self.inity = inity
        self.x = self.initx
        self.y = self.inity
        self.light = light
        self.name = 'sound'

    def act(self, a):
        if a==Actions.TOUCH and self.light.s == 1:
            if self.s == 0:
                self.s = 1
            else:
                self.s = 0
        elif a==Actions.TOUCHUP and self.light.s == 1 and self.s != 0:
            if self.s < 3:
                self.s += 1
        elif a==Actions.TOUCHDOWN and self.light.s == 1 and self.s != 0:
            if self.s > 0:
                self.s -= 1

    @property
    def smax(self):
        return 3

class Toy1(Obj):
    def __init__(self, light, initx=0, inity=0):
        super(Toy1, self).__init__()
        self.initx = initx
        self.inity = inity
        self.x = self.initx
        self.y = self.inity
        self.light = light
        self.name = 'toy1'

    def act(self, a):
        if a == Actions.TOUCH and self.light.s == 1:
            self.s = 1 - self.s
        elif a == Actions.TAKE:
            self.in_hand = 1
        elif a == Actions.PUT:
            self.in_hand = 0

    @property
    def smax(self):
        return 1

class Toy2(Toy1):
    def __init__(self, light, sound, initx=0, inity=0):
        super(Toy2, self).__init__(light, initx=initx, inity=inity)
        self.sound = sound
        self.name = 'toy2'

    def act(self, a):
        if a == Actions.TAKE:
            self.in_hand = 1
        elif a == Actions.PUT:
            self.in_hand = 0
        elif a == Actions.TOUCHDOWN and self.light.s == 1:
            if self.s == 0:
                self.s = 1
        elif a == Actions.TOUCHUP and self.light.s == 1:
            if self.s == 1:
                self.s = 2
        elif a == Actions.TOUCHLEFT and self.light.s == 1:
            if self.s == 2:
                self.s = 3
        elif a == Actions.TOUCHRIGHT and self.light.s == 1:
            if self.s == 3:
                self.s = 4
        elif a == Actions.TOUCH and self.light.s == 1:
            if self.s > 0:
                self.s = 0

    @property
    def smax(self):
        return 4

class Playroom(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')

        self.x = 0
        self.y = 0
        light = Light(initx=3, inity=4)
        sound = Sound(light=light, initx=5, inity=1)
        obj1 = Toy1(light=light, initx=2, inity=2)
        obj2 = Toy2(light=light, sound=sound, initx=1, inity=5)

        self.objects = [light, sound, obj1, obj2]
        self.lastaction = None
        self.maxR = 6
        self.maxC = 6

    def step(self, a):

        # print(self.get_state(), a)

        if a==Actions.UP:
            self.y = min(self.y + 1, self.maxR)
            h = self.get_held()
            if h >= 0:
                self.objects[h].y = self.y

        elif a==Actions.DOWN:
            self.y = max(self.y - 1, 0)
            h = self.get_held()
            if h >= 0:
                self.objects[h].y = self.y

        elif a==Actions.LEFT:
            self.x = max(self.x - 1, 0)
            h = self.get_held()
            if h >= 0:
                self.objects[h].x = self.x

        elif a==Actions.RIGHT:
            self.x = min(self.x + 1, self.maxC)
            h = self.get_held()
            if h >= 0:
                self.objects[h].x = self.x

        # elif a==Actions.TAKE:
        #     h = self.get_held()
        #     o = self.get_underagent()
        #     if o >= 0 and h == -1:
        #         self.objects[o].act(a)
        #
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

        return (self.get_state(), 0, False, {"prob" : 1})

    def get_underagent(self, x=0, y=0):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x + x and obj.y == self.y + y and obj.in_hand == 0:
                return i
        return -1

    def get_state(self):
        res = [self.x, self.y]
        for obj in self.objects:
            res += [obj.x, obj.y, obj.s, obj.in_hand]
        return res

    def get_held(self):
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
        return self.get_state()

    @property
    def state_high(self):
        smax = [self.maxR , self.maxC]
        for o in self.objects:
            smax += [self.maxR, self.maxC, o.smax, 1]
        return smax

    @property
    def state_init(self):
        sinit = [0, 0]
        for o in self.objects:
            sinit += [o.initx, o.inity, 0, 0]
        return sinit

    @property
    def state_low(self):
        return [0] * (2 + len(self.objects) * 4)

if __name__ == '__main__':
    env = Playroom()
    env.reset()
    for i in range(100000):
        while True:
            a = np.random.randint(11)
            if a != 4 and a != 5: break
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
