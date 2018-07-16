You will mostly just need to run simulation.py and plotting.py

In simulation.py you can change forest size, length of simulation, and adjust the number of simulations to run in the batch. When this is run an errors.npy and times.npy data files will be saved, with all of the raw data.

Plotting.py is just a small script we can modify to plot as desired, it loads errors.npy and times.npy.

factor_graph.py and factors.py are implementations of factor graph related utilities that I got from Karen Leung in my lab, from a class at Stanford. I have modified them for our needs.

lbp.py has the main part of the LBP code, that builds the graph and handles things at a higher level.

forest_fire_dynamics.py is has all the code that simulation.py uses to simulated the ground truth forest fire dynamics.
