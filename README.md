# estimate-large-GMDPs

A repository for methods to perform state estimation for graph-based MDPs. All methods are applied to a stochastic grid-based forest fire process. 

## Files and directories:
- `lbp`: Loopy Belief Propagation
- `FireSimulator.py`: Simulates a stochastic grid-based forest fire process
- `FireSimulatorUtilities.py`
- `energy minimization.ipynb`: Free-energy minimization method that is deterministic 
- `exact filter.ipynb`: Exact Bayes filter 
- `message passing.ipynb`: Message-passing algorithm based on the variational inference framework 
- `node bayes filter.ipynb`: Experimental method where a Bayes filter is used for each node
- `scratch inference.ipynb`: Code to investigate using variational inference