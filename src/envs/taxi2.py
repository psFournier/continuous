import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+-----------------+",
    "| : | : : : : : | |",
    "| : : : : : : : | |",
    "| : | : : : | : | |",
    "| : | : : : | : | |",
    "| : | : : : | : | |",
    "| : : : : : | : | |",
    "| : | : : : | : : |",
    "| : | : : : | : : |",
    "| : | : : : | : : |",
    "+-----------------+",
]

class Taxi2Env(discrete.DiscreteEnv):
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

        nS = (9**4)*2
        self.nR = 9
        self.nC = 9
        maxR = self.nR-1
        maxC = self.nC-1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        initial_state = self.encode(5, 5, 0, 0, 0)
        isd[initial_state] = 1
        for taxirow in range(self.nR):
            for taxicol in range(self.nC):
                for passrow in range(self.nR):
                    for passcol in range(self.nC):
                        for status in range(2):
                            state = self.encode(taxirow, taxicol, passrow, passcol, status)
                            for a in range(nA):
                                newtaxirow, newtaxicol, newpassrow, newpasscol, newstatus = \
                                    taxirow, taxicol, passrow, passcol, status
                                reward = 0
                                done = False

                                if a==0:
                                    newtaxirow = min(taxirow+1, maxR)
                                    if newstatus==1: newpassrow = newtaxirow
                                elif a==1:
                                    newtaxirow = max(taxirow-1, 0)
                                    if newstatus==1: newpassrow = newtaxirow

                                if a==2 and self.desc[1+taxirow,2*taxicol+2]==b":":
                                    newtaxicol = min(taxicol+1, maxC)
                                    if newstatus == 1: newpasscol = newtaxicol
                                elif a==3 and self.desc[1+taxirow,2*taxicol]==b":":
                                    newtaxicol = max(taxicol-1, 0)
                                    if newstatus == 1: newpasscol = newtaxicol

                                elif a==4 and status==0 and taxicol == passcol and taxirow == passrow:
                                    newstatus = 1

                                elif a==5 and status==1:
                                    newstatus = 0

                                newstate = self.encode(newtaxirow, newtaxicol, newpassrow, newpasscol, newstatus)
                                P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passrow, passcol, aboard):
        i = taxirow
        i *= self.nR
        i += taxicol
        i *= self.nC
        i += passrow
        i *= self.nR
        i += passcol
        i *= 2
        i += aboard
        return i

    def decode(self, i):
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % 9)
        i = i // 9
        out.append(i % 9)
        i = i // 9
        out.append(i % 9)
        i = i // 9
        out.append(i)
        assert 0 <= i < 9
        return reversed(out)

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
