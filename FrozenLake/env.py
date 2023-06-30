#  WARNING : this version of the environment only works when there is only one starting state S in the map
#  Need updates to allow several starting states
#  This Frozen Lake environment is inspired by: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
from gymnasium import Env
from gymnasium import utils
from gymnasium.spaces import Discrete
import numpy as np

#  Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#  Set of maps
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],

    "6x6": [
        "SFFFFF",
        "FFHFFF",
        "FFHFFF",
        "FFFHHF",
        "FFFFFF",
        "HFFFFG"
    ],

    "7x7": [
        "SFFFFFH",
        "FFFFFFH",
        "FHHFFFH",
        "FHHFFFH",
        "FFFFFFF",
        "FFFFFFF",
        "HHHHFFG"
    ],

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],

    "10x10": [
        "HFHFFHHGHH",
        "FFFHFFFFFF",
        "FHFFFFFHHF",
        "HFFFHHFFFF",
        "FFHFHFHFFH",
        "FFHFFFFFHH",
        "FHFFFHHFFH",
        "SFFFFFFHFF",
        "FFFHHFFFFH",
        "HFFFFFHFHH"
    ]

}

#  Class used for coloring the current state in the render function
class bcolors:
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

class MyFrozenLake(Env):

    def __init__(self, map_name="4x4", is_slippery=True, slip_probas=[1/3, 1/3, 1/3]):
        #  Map
        desc = MAPS[map_name]
        self.desc = np.asarray(desc, dtype="c")
        #  Action space
        self.action_space = Discrete(4)
        #  Dimension of the map
        self.nRow, self.nCol = self.desc.shape
        # Number of Actions, States
        nA, nS = 4, self.nRow*self.nCol
        #  Initial state
        self.state = self.init_state()
        #  State space
        self.observation_space = Discrete(self.nRow*self.nCol)
        # Last action (useful for the render)
        self.lastaction = None
        # Probabilities to slip
        self.slip_probas = slip_probas
        #  Probability matrix for the agent
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        #  Get the goal position (only one goal allowed)
        row_goal, col_goal = None, None
        for i in range(len(self.desc)):
            for j in range(len(self.desc[0])):
                if bytes(self.desc[i, j]) in b"G":
                    row_goal, col_goal = i, j
        self.goal_position = [row_goal, col_goal]

        #  Update the probability matrix
        #  Input: coordinates (int), action (int), and the possible future action (int)
        #  Last argument only used for wind probability matrix
        #  Output: new state (int), reward (float) and end of episode (boolean)
        def update_probability_matrix(row, col, action, future_action=None):
            newrow, newcol = self.inc(row, col, action)
            if future_action is not None:
                newstate = self.to_s(newrow, newcol) * nA + future_action
            else:
                newstate = self.to_s(newrow, newcol)
            newletter = self.desc[newrow, newcol]
            done = bytes(newletter) in b"GH"
            #  Change reward
            reward = float(newletter == b"G")

            return newstate, reward, done

        # Fill the probability matrix
        for row in range(self.nRow):
            for col in range(self.nCol):
                s = self.to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            i = 0
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (self.slip_probas[i], *update_probability_matrix(row, col, b))
                                )
                                i += 1
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        return

    #  From coordinates to state
    #  Input: coordinates (int)
    #  Output: a state (int)
    def to_s(self, row, col):
        return row * self.nCol + col

    #  Update coordinates
    #  Input: coordinates (int) and action (int)
    #  Output: new coordinates (int)
    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nRow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.nCol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return (row, col)

    #  The agent performs a step in the environment and arrive in a particular state
    #  according to the probability matrix, the current state, the action of both the agent and the wind
    #  Input: action (int), index of the action (int), already chosen new state (int) and P-scenario use (bool)
    #  Output: new state (int), the reward (int), done flag (bool) and probability
    #  to come into the new state (float)
    def step(self, action, new_state=None, p_scenario=False):
        #  Case of the agent's step
        transitions = self.P[self.state][action]
        #  Case of P-scenario with different transition probabilities
        if p_scenario and self.slip_probas.count(self.slip_probas[0]) != len(self.slip_probas):
            idx_best_transition = np.argmax(self.slip_probas)  # works in the case a single best proba
            p, s, r, d = transitions[idx_best_transition]
        #  Case of a simple agent's interaction
        else:
            #  Random choice among possible transitions
            i = np.random.choice(len(transitions), p=self.slip_probas)
            p, s, r, d = transitions[i]

        #  Updates
        self.state = s
        self.lastaction = action
        return (s, r, d, p)

    #  Reset the environment
    #  Input: None
    #  Output: initial state (int)
    def reset(self):
        self.state = self.init_state()
        self.lastaction = None
        return self.state

    #  Display to the user the current state of the environment
    #  Input:  None
    #  Output: None
    def render(self):

        #  Print the current action
        if self.lastaction != None:
            print("    ({})".format(["Left", "Down", "Right", "Up"][self.lastaction]))

        #  Get the current position
        row, col = self.state // self.nCol, self.state % self.nCol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        #  Highlight current position in red
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        #  Render
        for line in range(self.nRow):
            row_str = ""
            for column in range(self.nCol):
                row_str = row_str + desc[line][column]
            print(row_str)

        return

    #  Set a state
    #  Input: state (int)
    #  Output: None
    def setObs(self, obs):
        self.state = obs
        self.lastaction = None
        return

    #  Initialize the agent's state
    #  Input: None
    #  Output: initial state (int)
    def init_state(self):
        for i in range(len(self.desc)):
            for j in range(len(self.desc[0])):
                if bytes(self.desc[i, j]) in b"S":
                    return self.to_s(i, j)
