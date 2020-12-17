# filter-large-GHMMs

A repository to support the paper, **Scalable Filtering of Large Graph-coupled Hidden Markov Models**.

Paper citation:
```
@InProceedings{9029382,
  author={R. N. {Haksar} and J. {Lorenzetti} and M. {Schwager}},
  booktitle={2019 IEEE 58th Conference on Decision and Control (CDC)}, 
  title={Scalable Filtering of Large Graph-Coupled Hidden Markov Models}, 
  year={2019},
  pages={1307-1314},}
```

## Requirements:
- Developed with Python 3.6
- Requires `numpy` (version 1.16.0+)
- Requires the [simulators](https://github.com/rhaksar/simulators) repository

## Directories:
- `filters`: Directory with common functions and implementations of loopy belief propagation (LBP) and relaxed
  anonymous variational inference (RAVI). 

## Files:
- `exact_filter.py`: Recursive Bayesian Filter for the LatticeForest simulator.
- `lbpLatticeForest.py`: LBP for the LatticeForest simulator.
- `lbpWestAfrica.py`: LBP for the WestAfrica simulator. 
- `raviLatticeForest.py`: RAVI for the LatticeForest simulator.
- `raviWestAfrica.py`: RAVI for the WestAfrica simulator. 
