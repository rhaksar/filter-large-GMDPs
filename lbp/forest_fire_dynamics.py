import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import pdb

class tree:
    """ A single tree:
    Inputs:
    pos - position, an integer grid location (array)
    N - total number of trees in the current forest """

    def __init__(self, pos, W, H):
        """ states:
        0 = healthy
        1 = on fire 
        2 = burnt"""
        self.p = np.array(pos) # [row, col]
        self.state = 0
        self.alpha = 0.2763
        self.beta = np.exp(-1./10)

        # Create neighbor set
        neighbors = [np.array([pos[0] + 1, pos[1]]), # bottom (increasing row)
             np.array([pos[0], pos[1] + 1]), # right (increasing col)
             np.array([pos[0] - 1, pos[1]]), # top (decreasing row)
             np.array([pos[0], pos[1] - 1])] # left (decreasing col)

        if pos[1] == 0:
            del neighbors[3]
        if pos[0] == 0:
            del neighbors[2]
        if pos[1] == W-1:
            del neighbors[1]
        if pos[0] == H-1:
            del neighbors[0]

        self.n = neighbors


    def query_neighbors(self, current):
        states = []
        for i, [x,y] in enumerate(self.n):
            states.append(current[x][y])
        self.nstates = np.array(states)


    def update(self, current):
        """ Update based on Markov process """

        # Look at current forest to see state of neighbors
        self.query_neighbors(current)

        # If the tree is healthy
        if self.state == 0:
            # Find number of tree in neighbors that are on fire
            onfire = np.where(self.nstates==1)[0]
            f = onfire.size

            if np.random.rand() < 1 - self.alpha**f:
                self.state = 1

        # If the tree is already on fire
        elif self.state == 1:
            if np.random.rand() > self.beta:
                self.state = 2



class forest:
    """ The forest is modelled for now as a grid, with each tree being
    at an integer grid location from (0,0) to (N,N) """

    def __init__(self, W, H):
        self.forest = [[tree([row,col],W,H) for col in range(W)] for row in range(H)]
        self.W = W
        self.H = H
        self.seed_fire() # Start the fire at a location near middle
        self.query_state()

        self.end = False # indicates if simulation has terminated
                         # True when no fire

        # Create data structure to hold information
        col = np.array([range(self.W) for i in range(self.H)]).flatten()
        row = np.array([[i]*self.W for i in range(self.H)]).flatten()
        self.data = {'x':row,'y':col,'state':np.array(self.state).flatten()}

        # Add extra initial set to data, because filters need on step to update their first
        self.data['state'] = np.vstack((self.data['state'],np.array(self.state).flatten()))


    def reset(self):
        """ Resets the forest to original state, 
        with all trees healthy """
        for row in range(self.H):
            for col in range(self.W):
                self.forest[row][col].state = 0
        self.end = False
        self.seed_fire()


    def seed_fire(self):
        # """ Seeds a single tree to be one fire """
        row = int(self.H/2)
        col = int(self.W/2)
        self.forest[row][col].state = 1

        # """ Seeds a grid of trees on fire """
        # row_center = int(math.ceil(self.H/2))
        # col_center = int(math.ceil(self.W/2))
        # deltas = [k for k in range(-1,3)]
        # deltas = itertools.product(deltas,deltas)
        #
        # for (drow,dcol) in deltas:
        #     row = row_center + drow
        #     col = col_center + dcol
        #     self.forest[row][col].state = 1

        self.query_state()


    def advance(self):
        """ Simulates one time step """

        # if no more fire, return
        if self.end:
            print 'process has terminated'
            return

        for row in range(self.H):
            for col in range(self.W):
                self.forest[row][col].update(self.state)
        self.query_state()

        # Append new state to the history
        s = np.array(self.state).flatten()
        self.data['state'] = np.vstack((self.data['state'],s))

        # check if no more fires
        if np.any(np.array(self.state) == 1):
            self.end = False
        else:
            self.end = True

    def query_state(self):
        """ Generates matrix of current forest state """
        self.state = [[self.forest[row][col].state for col in range(self.W)] 
                            for row in range(self.H)]


    def query_state_locations(self,i,sval):
        """ Returns the x and y vectors for the given state 
        Inputs:
        i: time step index
        sval: 0,1,2 """

        # Crap because I can't code well
        if self.data['state'].size > self.H*self.W and i <= self.data['state'].shape[0] - 1:
            idx = np.where(self.data['state'][i]==sval)[0]
            x = self.data['x'][idx]; y = self.data['y'][idx]
        elif i == 0:
            idx = np.where(self.data['state']==sval)[0]
            x = self.data['x'][idx]; y = self.data['y'][idx]            
        else:
            plt.close('all')
            sys.exit('Index too large.')
        return y,x



