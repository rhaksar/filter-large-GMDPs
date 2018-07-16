###############################################################################
# factor graph data structure implementation 
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

from factors import *
import numpy as np
import pdb

class FactorGraph:
    def __init__(self,  W, H, numVar=0, numFactor=0):
        '''
        var list: index/names of variables

        domain list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]

        varToFactor: list of lists, it has the same length as the number of variables. 
                     varToFactor[i] is a list of the indices of Factors that are connected to variable i

        factorToVar: list of lists, it has the same length as the number of factors. 
                     factorToVar[i] is a list of the indices of Variables that are connected to factor i

        factors: a list of Factors

        messagesVarToFactor: a dictionary to store the messages from variables to factors,
                            keys are (src, dst), values are the corresponding messages of type Factor

        messagesFactorToVar: a dictionary to store the messages from factors to variables,
                            keys are (src, dst), values are the corresponding messages of type Factor
        '''
        self.W = W
        self.H = H
        self.var = []
        self.varName = [[] for _ in range(numVar)]
        self.domain = [[0,1] for _ in range(numVar)]
        self.varToFactor = [[] for _ in range(numVar)]
        self.factorToVar = [[] for _ in range(numFactor)]
        self.factors = []
        self.facName = []
        self.messagesVarToFactor = {}
        self.messagesFactorToVar = {}
        self.layerHist = []
    
    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigments
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factors:
            output *= f.val.flat[assignment_to_indices([a[f.scope]], f.card)]
        return output[0]
    
    def getInMessage(self, src, dst, type="varToFactor"):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        param - type: type of messages. "varToFactor" is the messages from variables to factors; 
                    "factorToVar" is the message from factors to variables
        return: message from src to dst
        
        In this function, the message will be initialized as an all-one vector (normalized) if 
        it is not computed and used before. 
        '''
        if type == "varToFactor":
            if (src, dst) not in self.messagesVarToFactor:
                inMsg = Factor()
                inMsg.scope = [src]
                inMsg.card = [len(self.domain[src])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesVarToFactor[(src, dst)] = inMsg
            return self.messagesVarToFactor[(src, dst)]

        if type == "factorToVar":
            if (src, dst) not in self.messagesFactorToVar:
                inMsg = Factor()
                inMsg.scope = [dst]
                inMsg.card = [len(self.domain[dst])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesFactorToVar[(src, dst)] = inMsg
            return self.messagesFactorToVar[(src, dst)]


    def runParallelLoopyBP(self, maxIterations): 
        '''
        param - iterations: the number of iterations you do loopy BP
          
        In this method, you need to implement the loopy BP algorithm. The only values 
        you should update in this function are self.messagesVarToFactor and self.messagesFactorToVar. 
        
        Warning: Don't forget to normalize the message at each time. You may find the normalize
        method in Factor useful.
        '''      
        ###############################################################################
        numVariables = len(self.var)
        xhatPrev = np.zeros(numVariables)
        xhat = np.zeros(numVariables)

        # First initialize all messages to and from each factor
        for (i,f) in enumerate(self.factors):
            scope = f.scope
            for s in scope:
                # factor to scope
                self.getInMessage(i, s, type="factorToVar")
                self.messagesFactorToVar[(i,s)].name = "[factorToVar_%i_%i]"%(i,s)
                self.getInMessage(s, i, type="varToFactor")
                self.messagesVarToFactor[(s,i)].name = "[varToFactor_%i_%i]"%(i,s)

        # Then go through iterations of LBP
        for it in range(maxIterations):
            # Variable to factor messages
            for k in self.messagesVarToFactor:
                var, fac = k # message from var to fac
                nu = Factor() # actual message to pass is a factor

                # For all factors connected to var that is not fac, create list
                neighbour_f = [f for f in self.varToFactor[var] if f != fac]

                # Build messages from each individual other factor into list
                mus = [self.getInMessage(t, var, type="factorToVar") for t in neighbour_f]
                
                # Then multiply, as per message passing rules
                for mu in mus:
                    nu = nu.multiply(mu)

                # Set message from var to fac as the normalized product
                self.messagesVarToFactor[(var, fac)] = nu.normalize()

            # Now do factor to variable messages
            for k in self.messagesFactorToVar:
                fac, var = k
                mu = Factor() # actual message to variable

                # Get all neighbor variables of the factor fac
                neighbour_v = [v for v in self.factorToVar[fac] if v != var] 
                nus = [self.getInMessage(j, fac, type="varToFactor") for j in neighbour_v]
                # Product of all incoming messages
                for nu in nus:
                    mu = mu.multiply(nu)

                # Get the actual factor
                fs = self.factors[fac]

                # Marginalize over all variables in argument except var
                mu = mu.multiply(fs)
                mu = mu.marginalize_all_but([var])
                self.messagesFactorToVar[(fac,var)] = mu.normalize()

            # Compute marginal at the end
            for (i, varIdx) in enumerate(self.var):
                belief = self.estimateMarginalProbability(varIdx)
                xhat[i] = np.argmax(belief)

            # Break if less than 1% of nodes are changing estimate at each iteration
            change = float(np.where(xhat != xhatPrev)[0].size)
            # print change/numVariables
            if change/numVariables <= .01:
                break

            xhatPrev = np.copy(xhat)


    def estimateMarginalProbability(self, var):
        '''
        Estimate the marginal probabilities of a single variable after running 
        loopy belief propogation.  (This method assumes runParallelLoopyBP has
        been run)

        param - var: a single variable index
        return: numpy array of size 2 containing the marginal probabilities 
                that the variable takes the values 0 and 1
        
        example: 
        >>> factor_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]
    
        Since in this assignment, we only care about the marginal 
        probability of a single variable, you only need to implement the marginal 
        query of a single variable.     
        '''
        ###############################################################################
        # To do: your code here  
        # get neighbours of var
        # self.varToFactor
        neighbour_messages = [self.getInMessage(f, var, type="factorToVar") for f in self.varToFactor[var]]
        p = Factor()
        for nm in neighbour_messages:
            p = p.multiply(nm)
        p = p.normalize()
        return p.val
 
        ###############################################################################
    

    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP 
        assignments for the variables. You may utilize the method
        estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        >>> factor_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''
        
        output = np.zeros(len(self.var))
        ###############################################################################
        # To do: your code here  
        for (i,v) in enumerate(self.var):
            p = self.estimateMarginalProbability(v)
            output[i] = 1*(p[0] < p[1])

        ###############################################################################  
        return output


    def node2index(self, row, col, t):
        return t*self.W*self.H + row*self.W + col
