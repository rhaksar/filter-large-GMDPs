# filter-large-GHMMs

A repository of methods for filtering graph-based MDPs.  

## Requirements:
- Developed with Python 3.5
- Requires `numpy` (version 1.16.0+)
- Requires the [simulators](https://github.com/rhaksar/simulators) repository: clone the repository into the root level of this repository 

## Files:
- `factor_graph.py`: Helper code to build factor graphs.
- `factors.py`: Factor definition and operations for factor graphs.
- `lbp.py`: General implementation of Loopy Belief Propagation (LBP) for filtering.
- `lbpLatticeForest.py`: Application of LBP to the LatticeForest simulator.
- `lbpWestAfrica.py`: Application of LBP to the WestAfrica simulator. 
- `Observe.py`: Defines measurement models for filters. 
- `ravi.py`: General implementation of Relaxed Anonymous Variational Inference (RAVI) for filtering. 
- `raviLatticeForest.py`: Application of RAVI to the LatticeForest simulator.
- `raviWestAfrica.py`: Application of RAVI to the WestAfrica simulator. 