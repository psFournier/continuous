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
    "|  _ _  | |",
    "| |  _  | |",
    "| | | | | |",
    "| |   |_| |",
    "|_ _|_ _ _|",
    ]

class Taxi1(discrete.DiscreteEnv):
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
        self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]
        maxR = self.nR-1
        maxC = self.nC-1
        nS = self.nR * self.nC * 5
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        initial_state = self.encode(0, 0, 2)
        isd[initial_state] = 1
        for taxirow in range(self.nR):
            for taxicol in range(self.nC):
                for status in range(5):
                        state = self.encode(taxirow, taxicol, status)
                        for a in range(nA):
                            newtaxirow, newtaxicol, newstatus = taxirow, taxicol, status

                            if a == 0 and self.desc[1 + taxirow, 1 + 2 * taxicol] == b" ":
                                newtaxirow = min(taxirow+1, maxR)

                            elif a == 1 and self.desc[taxirow, 1 + 2 * taxicol] == b" ":
                                newtaxirow = max(taxirow-1, 0)

                            elif a == 2 and self.desc[1 + taxirow, 2 * taxicol + 2] == b" ":
                                newtaxicol = min(taxicol+1, maxC)

                            elif a == 3 and self.desc[1 + taxirow, 2 * taxicol] == b" ":
                                newtaxicol = max(taxicol-1, 0)

                            elif a == 4 and status < 4 and (taxirow, taxicol) == self.locs[status]:
                                newstatus = 4
                            elif a == 5 and status == 4 and (taxirow, taxicol) in self.locs:
                                    newstatus = self.locs.index((taxirow, taxicol))

                            newstate = self.encode(newtaxirow, newtaxicol, newstatus)
                            P[state][a].append((1.0, newstate, 0, False))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    @property
    def nR(self):
        return int(self.desc.shape[0] - 1)

    @property
    def nC(self):
        return int((self.desc.shape[1] - 1) / 2)

    def encode(self, taxirow, taxicol, passloc):
        # (5) 5, 5
        i = taxirow
        i *= self.nC
        i += taxicol
        i *= 5
        i += passloc
        return i

    def decode(self, i):
        out = []
        out.append(i % 5)
        i = i // 5
        out.append(i % self.nC)
        i = i // self.nC
        out.append(i)
        assert 0 <= i < self.nR
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, status = self.decode(self.s)
        if status < 4:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[status]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'magenta', highlight=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'green', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pick", "Drop"][self.lastaction]))
        else: outfile.write("\n")
        # No need to return anything for human
        if mode != 'human':
            return outfile

if __name__ == '__main__':
    env = Taxi1()
    env.reset()
    for _ in range(100000):
        env.render()
        a = np.random.randint(6)
        env.step(a)