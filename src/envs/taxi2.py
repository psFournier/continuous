import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

# MAP = [
#     " _ _ _ _ _ _ _ _ _ ",
#     "|  _ _  |  _ _  | |",
#     "| |   |  _|     | |",
#     "| | |  _ _ _| | | |",
#     "| |_|  _ _|   | | |",
#     "|_  |_ _|   | | | |",
#     "|    _ _| |_| | | |",
#     "| | |    _  | | | |",
#     "| |_|  _| | |_|_| |",
#     "|_ _ _ _ _|_ _ _ _|",
#     ]

MAP = [
    " _ _ _ _ _ ",
    "|       | |",
    "|    _ _| |",
    "|  _ _  | |",
    "|_ _    | |",
    "|_ _ _ _ _|",
    ]

class Taxi2(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "œœœ"
    by Tom Dietterich

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        maxR = self.nR-1
        maxC = self.nC-1
        nS = ((self.nR * self.nC) ** 2) * 2
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for taxirow in range(self.nR):
            for taxicol in range(self.nC):
                for passrow in range(self.nR):
                    for passcol in range(self.nC):
                        for status in range(2):
                            state = self.encode(taxirow, taxicol, passrow, passcol, status)
                            if taxirow == 0 and taxicol == 0 and status == 0:
                                isd[self.encode(taxirow, taxicol, passrow, passcol, status)] = 1
                            for a in range(nA):
                                newtaxirow, newtaxicol, newpassrow, newpasscol, newstatus = \
                                    taxirow, taxicol, passrow, passcol, status

                                if a == 0 and self.desc[1 + taxirow, 1 + 2 * taxicol] == b" ":
                                    newtaxirow = min(taxirow+1, maxR)
                                    if newstatus==1: newpassrow = newtaxirow

                                elif a == 1 and self.desc[taxirow, 1 + 2 * taxicol] == b" ":
                                    newtaxirow = max(taxirow-1, 0)
                                    if newstatus==1: newpassrow = newtaxirow

                                elif a == 2 and self.desc[1 + taxirow, 2 * taxicol + 2] == b" ":
                                    newtaxicol = min(taxicol+1, maxC)
                                    if newstatus == 1: newpasscol = newtaxicol

                                elif a == 3 and self.desc[1 + taxirow, 2 * taxicol] == b" ":
                                    newtaxicol = max(taxicol-1, 0)
                                    if newstatus == 1: newpasscol = newtaxicol

                                elif a == 4 and status==0 and taxicol == passcol and taxirow == passrow:
                                    newstatus = 1

                                elif a == 5 and status==1:
                                    newstatus = 0

                                newstate = self.encode(newtaxirow, newtaxicol, newpassrow, newpasscol, newstatus)
                                P[state][a].append((1.0, newstate, 0, False))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def step(self, action):
        obs, _, _, _ = super(Taxi2, self).step(action)
        state = np.array(list(self.decode(obs)))
        return state

    def reset(self):
        obs = super(Taxi2, self).reset()
        state = np.array(list(self.decode(obs)))
        return state

    @property
    def nR(self):
        return int(self.desc.shape[0] - 1)

    @property
    def nC(self):
        return int((self.desc.shape[1] - 1) / 2)

    def encode(self, taxirow, taxicol, passrow, passcol, aboard):
        i = taxirow
        i *= self.nC
        i += taxicol
        i *= self.nR
        i += passrow
        i *= self.nC
        i += passcol
        i *= 2
        i += aboard
        return i

    def decode(self, i):
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % self.nC)
        i = i // self.nC
        out.append(i % self.nR)
        i = i // self.nR
        out.append(i % self.nC)
        i = i // self.nC
        out.append(i)
        assert 0 <= i < self.nR
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passrow, passcol, status = self.decode(self.s)
        if status == 0:
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(out[1 + taxirow][2 * taxicol + 1], 'yellow',
                                                               highlight=True)
            out[1 + passrow][2 * passcol + 1] = utils.colorize(out[1 + passrow][2 * passcol + 1], 'magenta',
                                                               highlight=True)
        elif status == 1:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'green', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pick", "Drop"][self.lastaction]))
        else: outfile.write("\n")
        # No need to return anything for human
        if mode != 'human':
            return outfile

if __name__ == '__main__':
    env = Taxi2()
    env.reset()
    for _ in range(100000):
        env.render()
        a = np.random.randint(6)
        env.step(a)
