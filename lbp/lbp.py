import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *
import pdb

class LBPFilter:
    """ The loopy belief propagation filter for a forest with size
    (W, H) """

    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.alpha = 0.2763 # forest fire spread parameter
        self.beta = np.exp(-1./10) # fire persistence parameter

        # Initialize graph
        self.G = FactorGraph(W, H, numVar=0, numFactor=0)

        # Some other useful variables
        self.factorIdx = 0
        self.t = 0


    def addNewLayer(self, y, prior = None):
        """ Adds a new time layer to the graph corresponding to the new
        measurement y. If prior is given (list of lists of numpy arrays)
        and self.t = 0 then the layer also includes factors corresponding
        to the priors. First it builds the nodes for each new tree at the
        current time, then it adds factors for measurements. Then if
        self.t > 0 it adds factors for the dynamics. """

        # Add proper room for factors and variables
        treeNodes = self.W*self.H # add this many new nodes for trees
        if self.t == 0 and prior is None: # don't need to add extra factors
                treeFactors = 0
        else: # factors for either priors (first step) or transitions
            treeFactors = self.W*self.H
        yFactors = self.W*self.H # factors for measurements
        factorCount = treeFactors + yFactors # total factors to add
        self.G.varName += [[] for _ in range(treeNodes)]
        self.G.domain += [[0,1,2] for _ in range(treeNodes)]
        self.G.varToFactor += [[] for _ in range(treeNodes)]
        self.G.factorToVar += [[] for _ in range(factorCount)]

        # On this layer add variables and factors (for each tree)
        for row in range(self.H):
            for col in range(self.W):
                # Get variable index and name for this factor
                varIdx = self.node2index(row, col, self.t)
                xname = 'x' + str(row) + str(col) + str(self.t)

                self.G.var.append(varIdx)
                self.G.varName[varIdx] = xname

                # Create factor for measurement y_t^rc to x_t^rc
                scope=[varIdx] # scope is simply xrct
                card = [3] # both x and y nodes can only take on three values (0,1,2), so both have cardinality 3
                val = np.zeros(shape=tuple(card)) # val is array of shape card where each element corresponds to the factor value with those inputs
                for x in [0,1,2]:
                    if x == y[row, col]:
                        val[x] = 0.9 # this means when X=x and Y=y, the value of the factor is - (measurement model)
                    else:
                        val[x] = 0.05

                name = 'g_%i%i%i' % (row,col,self.t)
                self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                # varToFactor[i] is list of indices of factors corresponding to variable i
                # factorToVar[j] is list of variables connected to factor j
                self.G.varToFactor[varIdx] += [self.factorIdx]
                self.G.factorToVar[self.factorIdx] += [varIdx] # only one variable connected to each measurement variable
                self.factorIdx += 1

                # In some cases we want to add additional factors to the first
                # set of nodes that represent the prior beliefs
                if self.t == 0 and prior is not None:
                    # Create prior factors
                    scope=[varIdx] # scope is simply xrct
                    card = [3] 
                    val = prior[row][col]

                    name = 'b_%i%i%i' % (row,col,self.t)
                    self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                    # varToFactor[i] is list of indices of factors corresponding to variable i
                    # factorToVar[j] is list of variables connected to factor j
                    self.G.varToFactor[varIdx] += [self.factorIdx]
                    self.G.factorToVar[self.factorIdx] += [varIdx]
                    self.factorIdx += 1

                # For all of the others, include factors in between time steps
                elif self.t > 0:
                    # Create factor from node x_t-1^rc to x_t^rc
                    neighbors = self.getNeighbors([row, col])
                    scope=[self.node2index(row, col, self.t-1)] # add self at t-1
                    for neighbor in neighbors: # add neighbors at t-1
                        scope += [self.node2index(neighbor[0], neighbor[1], self.t-1)]
                    scope += [varIdx] # add self at t

                    card = [3 for i in range(len(scope))]

                    val = np.zeros(shape=tuple(card))
                    iterate = [[0,1,2] for i in range(len(scope))]
                    for combo in itertools.product(*iterate):
                        # These must follow scope order (xt-1, xn1t-1, xn2t-1, .. xnmt-1, xt)
                        xtm1 = combo[0]
                        xt = combo[-1]
                        f = np.sum(combo[1:-1]) # sum neighbors states
                        if xtm1 == 0: # tree is currently healthy at t-1
                            if xt == 0: # tree remains healthy at time t
                                val[combo] = self.alpha**f
                            elif xt == 1: # tree becomes on fire at time t
                                val[combo] = 1 - self.alpha**f
                            else: # tree cannot go from healthy to burnt
                                val[combo] = 0.
                        elif xtm1 == 1: # tree is on fire at t-1
                            if xt == 0: # tree cannot be healthy at time t
                                val[combo] = 0.
                            elif xt == 1: # tree stays on fire
                                val[combo] = self.beta
                            else: # tree transitions to burnt at time t
                                val[combo] = 1 - self.beta
                        else: # tree is burnt at t-1
                            if xt == 0:
                                val[combo] = 0.
                            elif xt == 1:
                                val[combo] = 0.
                            else: # tree must remain burnt at time t
                                val[combo] = 1.

                    name = 'f_%i%i%i' % (row,col,self.t-1)

                    self.G.factors.append(Factor(scope=scope, card=card, val=val, name=name))

                    for idx in scope:
                        # Connect this new factor to each variable's list
                        self.G.varToFactor[idx] += [self.factorIdx]

                    # Connect this variable to factor's list
                    self.G.factorToVar[self.factorIdx] += scope
                    self.factorIdx += 1

        self.t += 1


    def varIdx2varfac(self, varIdx):
        var = self.G.var[varIdx]
        facIdxs = self.G.varToFactor[varIdx]
        fac = []
        for idx in facIdxs:
            fac += [self.G.factors[idx].name]
        return var, fac


    def fIdx2facvar(self, facIdx):
        fac = self.G.factors[facIdx].name
        varIdxs = self.G.factorToVar[facIdx]
        var = []
        for idx in varIdxs:
            var += [self.G.var[idx]]

        return fac, var


    def node2index(self, row, col, t):
        return t*self.W*self.H + row*self.W + col


    def index2node(self, idx):
        N = self.W*self.H
        t = idx/N
        rem = idx % N
        row = rem/self.W
        col = rem % self.W
        return row, col, t


    def getNeighbors(self, pos):
        # Create neighbor set
        neighbors = [np.array([pos[0] + 1, pos[1]]), # bottom (increasing row)
             np.array([pos[0], pos[1] + 1]), # right (increasing col)
             np.array([pos[0] - 1, pos[1]]), # top (decreasing row)
             np.array([pos[0], pos[1] - 1])] # left (decreasing col)

        if pos[1] == 0:
            del neighbors[3]
        if pos[0] == 0:
            del neighbors[2]
        if pos[1] == self.W-1:
            del neighbors[1]
        if pos[0] == self.H-1:
            del neighbors[0]

        return neighbors



class robot:
    """ This class is a wrapper to perform the high level implementation
    of Loopy Belief Propagation """

    def __init__(self, W, H, initialEstimate=None, maxIterations=10, horizon=5):
        self.W = W
        self.H = H
        N = W*H

        # Initialize graph
        self.graph = LBPFilter(W, H)
        self.maxIterations = maxIterations # for each LBP run

        # Initialize estimates
        # self.estimate = np.zeros((H, W))
        # self.confidence = 0.5*np.ones((H, W))
        if initialEstimate is None:
            self.estimate = np.zeros((H, W))
            self.confidence = 0.33*np.ones((H, W))
        else:
            self.estimate = initialEstimate
            self.confidence = np.ones((H, W))
        
        # To store data in
        col = np.array([range(self.W) for i in range(self.H)]).flatten()
        row = np.array([[i]*self.W for i in range(self.H)]).flatten()
        self.data = {'x':row,'y':col,'est':np.array(self.estimate).flatten(),
                        'lvl':np.array(self.confidence).flatten()}
        self.measurementHist = []

        # Horizon for LBP graph growth, after it has this many time steps
        # it will truncate the end to maintain this size (keeps computation
        # time to a constant in time)
        self.h = horizon


    def measure(self, noisyMeasurement):
        """ Takes a noisy measurement of the environment """
        self.measurement = noisyMeasurement
        self.measurementHist += [noisyMeasurement]


    def advance(self):
        """ Called at each time step to run LBP """

        if self.graph.t == 0: # Very beginning
            # Add a new layer to the filter for the new measurement/time
            self.graph.addNewLayer(self.measurement)
            self.estimate = self.measurement
            self.confidence = 0.9*np.ones((self.H, self.W))

        elif self.graph.t >= 1 and self.graph.t < self.h: # Still beginning, before we reach horizon
            # Add a new layer to the filter for the new measurement/time
            self.graph.addNewLayer(self.measurement)

            # Run LBP
            # print "Running LBP"
            self.graph.G.runParallelLoopyBP(self.maxIterations)

            self.query_estimate()

        else: # Build a new graph
            # Get priors from old graph before deleting
            priors = self.query_beliefs(1) # use t = 1

            # Get measurements for new graph
            measurements = self.measurementHist[-self.h:]

            # Create a fresh graph
            self.graph = LBPFilter(self.W, self.H)

            # Create the first layer, which has measurements and priors
            self.graph.addNewLayer(measurements[0], priors)

            # Create the rest of the layers
            for i in range(1, self.h):
                self.graph.addNewLayer(measurements[i])

            # print "Running LBP"
            self.graph.G.runParallelLoopyBP(self.maxIterations)

            self.query_estimate()

        # Append to data
        est = np.array(self.estimate).flatten()
        self.data['est'] = np.vstack((self.data['est'],est))
        this = np.array(self.estimate)

        lvl = np.array(self.confidence).flatten()
        self.data['lvl'] = np.vstack((self.data['lvl'],lvl))


    def query_estimate(self):
        """ Generates matrix of states of highest probability """

        for row in range(self.H):
            for col in range(self.W):
                varIdx = self.graph.node2index(row, col, self.graph.t-1)
                belief = self.graph.G.estimateMarginalProbability(varIdx)
                self.estimate[row, col] = np.argmax(belief)
                self.confidence[row, col] = np.max(belief)


    def query_beliefs(self, t):
        """ Generates matrix of states of highest probability """
        beliefs = []

        for row in range(self.H):
            rowBelief = []
            for col in range(self.W):
                varIdx = self.graph.node2index(row, col, t)
                belief = self.graph.G.estimateMarginalProbability(varIdx)
                rowBelief += [belief]
            beliefs += [rowBelief]

        return beliefs


    def query_est_locations(self,i,sval):
        """ Returns the x and y vectors for the given estimate
        Inputs:
        i: time step index
        sval: 0,1 """

        # Crap because I can't code well
        if self.data['est'].size > self.W*self.H and i <= self.data['est'].shape[0] - 1:
            idx = np.where(self.data['est'][i]==sval)[0]
            x = self.data['x'][idx]; y = self.data['y'][idx]
            s = self.data['lvl'][i][idx]
        elif i == 0:
            idx = np.where(self.data['est']==sval)[0]
            x = self.data['x'][idx]; y = self.data['y'][idx]
            s = self.data['lvl'][i][idx]          
        else:
            plt.close('all')
            sys.exit('Index too large.')
        return y,x,s


    def grid_to_array(self):
        """ Generates the vectors for plotting """
        # Create x vector
        x = [range(self.W) for i in range(self.W)]
        x = np.array(x).flatten()

        # Create y vector
        y = [[i]*self.H for i in range(self.H)]
        y = np.array(y).flatten()

        # Create state vector
        s = np.array(self.state).flatten()

        # Now create different vectors for different states to plot
        idxh = np.where(s==0)[0]
        xh = x[idxh]; yh = y[idxh]

        idxf = np.where(s==1)[0]
        xf = x[idxf]; yf = y[idxf]

        return xh,yh,xf,yf
