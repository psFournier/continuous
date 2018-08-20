import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    " _ _ _ _ _ _ _ _ _ ",
    "|  _ _  |  _ _  | |",
    "| |   |  _|     | |",
    "| | |  _ _ _| | | |",
    "| |_|  _ _|   | | |",
    "|_  |_ _|   | | | |",
    "|    _ _| |_| | | |",
    "| | |    _  | | | |",
    "| |_|  _| | |_|_| |",
    "|_ _ _ _ _|_ _ _ _|",
    ]

# MAP = [
#     " _ _ _ _ _ ",
#     "|  _ _  | |",
#     "| |  _  | |",
#     "| | | | | |",
#     "| |   |_| |",
#     "|_ _|_ _ _|",
#     ]

class Labyrinth(discrete.DiscreteEnv):
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
        nS = self.nR * self.nC
        self.destrow, self.destcol = maxR, maxC
        isd = np.zeros(nS)
        nA = 4
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        initial_state = self.encode(0, 0)
        isd[initial_state] = 1
        for row in range(self.nR):
            for col in range(self.nC):
                state = self.encode(row, col)
                for a in range(nA):
                    newrow, newcol = row, col

                    if a == 0 and self.desc[1 + row, 1 + 2 * col] == b" ":
                        newrow = min(row+1, maxR)

                    elif a == 1 and self.desc[row, 1 + 2 * col] == b" ":
                        newrow = max(row-1, 0)

                    elif a == 2 and self.desc[1 + row, 2 * col + 2] == b" ":
                        newcol = min(col + 1, maxC)

                    elif a == 3 and self.desc[1 + row, 2 * col] == b" ":
                        newcol = max(col - 1, 0)

                    newstate = self.encode(newrow, newcol)
                    P[state][a].append((1.0, newstate, 0, False))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    @property
    def nR(self):
        return int(self.desc.shape[0] - 1)

    @property
    def nC(self):
        return int((self.desc.shape[1] - 1) / 2)

    def encode(self, row, col):
        i = row
        i *= self.nC
        i += col
        return i

    def decode(self, i):
        out = []
        out.append(i % self.nC)
        i = i // self.nC
        out.append(i)
        assert 0 <= i < self.nR
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        row, col = self.decode(self.s)
        out[1 + row][2 * col + 1] = utils.colorize(out[1 + row][2 * col + 1], 'yellow', highlight=True)

        out[1+self.destrow][2*self.destcol+1] = utils.colorize(out[1+self.destrow][2*self.destcol+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile

if __name__ == '__main__':
    env = Labyrinth()
    env.reset()
    for _ in range(100000):
        env.render()
        a = np.random.randint(4)
        env.step(a)