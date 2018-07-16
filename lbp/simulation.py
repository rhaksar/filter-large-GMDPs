import forest_fire_dynamics as ff
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np
import pdb
import sys
import lbp
import time


def simulate(nsteps, forest):
    """ Perform simulation """

    # Make sure forest is reset to original state
    forest.reset()

    # Get first measurement
    measurement = forestMeasurement(forest)

    # Robot (that performs LBP)
    robot = lbp.robot(W, H)
    robot.measure(measurement)
    robot.advance()

    # Error
    errors = np.zeros((2, nsteps))
    compTime = np.zeros(nsteps-1)
    robotE = errorTotal(forest.state, robot.estimate)
    measurementE = errorTotal(forest.state, measurement)
    errors[:,0] = np.array([robotE, measurementE])
    print robotE, measurementE

    # Simulate, and calculate error at each time
    for i in range(nsteps - 1):
        forest.advance()
        measurement = forestMeasurement(forest)

        t0 = time.time()
        robot.measure(measurement)
        robot.advance()
        t1 = time.time()
        compTime[i] = t1 - t0

        # Compute error metric
        robotE = errorTotal(forest.state, robot.estimate)
        measurementE = errorTotal(forest.state, measurement)
        errors[:,i+1] = np.array([robotE, measurementE])
        print robotE, measurementE

    return errors, compTime


def error(true, belief, confidence):
    count = 0
    W = belief.shape[0]
    H = belief.shape[1]

    for i in range(W):
        for j in range(H):
            if confidence[i,j] > 0.75:
                count += np.abs(true[i,j] - belief[i,j])
    return count


def errorTotal(true, belief):
    return np.sum(np.array(true)!=belief)


def movingAverageEstimate(history, measurement):
    if history is None:
        history = [np.copy(measurement)]
        estimate = measurement
    elif len(history) < 3:
        history.append(measurement)
        estimate = measurement
    else:
        history.append(measurement)
        estimate = (history[-1]*.5 + history[-2]*.3 + history[-3]*.2)

    return np.round(estimate), history


def forestMeasurement(forest):
    measure_model = np.array([0.9, 0.05, 0.05]) # [correct, incorrect]
    state = np.array(forest.state)
    noisyMeasurement = np.zeros(state.shape)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if np.random.rand() < measure_model[0]: # We have taken a correct measurement
                noisyMeasurement[i,j] = state[i,j]
            else:
                possible = [0, 1, 2]
                possible.remove(state[i,j])
                if np.random.rand() < 0.5:
                    noisyMeasurement[i,j] = possible[0]
                else:
                    noisyMeasurement[i,j] = possible[1]

    return noisyMeasurement



def animation(forest,robot):
    # Number of frames
    W = robot.W
    H = robot.H
    n = forest.data['state'].shape[0] - 1

    # Get initial vectors
    xh,yh = forest.query_state_locations(0,0)
    xf,yf = forest.query_state_locations(0,1)

    xh_est,yh_est,sh = robot.query_est_locations(0,0)
    xf_est,yf_est,sf = robot.query_est_locations(0,1)

    # Create plot for the real states
    plt.close('all')
    fig = plt.figure()
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlim([-1,W])
    plt.ylim([-1,H])
    plt.xticks([])
    plt.yticks([])
    plt.title('True Forest State')
    h_real = ax1.scatter(xh,yh,c=np.ones(xh.size),s=50, lw=0,
        cmap=cm.Greens, vmin=0.5, vmax=1.1)
    f_real = ax1.scatter(xf,yf,c=np.ones(xf.size),s=50, lw=0,
        cmap=cm.OrRd, vmin=0.5, vmax=1.1)

    # Create plot for the estimates
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlim([-1,W])
    plt.ylim([-1,H])
    plt.xticks([])
    plt.yticks([])
    plt.title('Estimate: Full Forest Observation')
    h_est = ax2.scatter(xh_est, yh_est, c = sh, s=50, lw=0,
                cmap=cm.Greens, vmin=0.5, vmax=1.1)
    f_est = ax2.scatter(xf_est, yf_est, c = sf, s=50, lw=0,
                cmap=cm.OrRd, vmin=0.5, vmax=1.1)

    h = ani.FuncAnimation(fig,animation_update,frames=n,repeat=False,
        interval=1000,fargs=(forest,robot,h_real,f_real,h_est,f_est))
    h.save('estimate.gif', dpi=80, writer='imagemagick')
    plt.show()

def animation_update(i,forest,robot,h_real,f_real,h_est,f_est):
    xh,yh = forest.query_state_locations(i,0)
    xf,yf = forest.query_state_locations(i,1)
    xh_est,yh_est,sh = robot.query_est_locations(i,0)
    xf_est,yf_est,sf = robot.query_est_locations(i,1)

    h_real.set_offsets(
             np.hstack((xh[:,np.newaxis],yh[:,np.newaxis])))
    h_real.set_array(np.ones(xh.size))

    f_real.set_offsets(
            np.hstack((xf[:,np.newaxis],yf[:,np.newaxis])))
    f_real.set_array(np.ones(xf.size))

    h_est.set_offsets(
             np.hstack((xh_est[:,np.newaxis],yh_est[:,np.newaxis])))
    h_est.set_array(sh)

    f_est.set_offsets(
            np.hstack((xf_est[:,np.newaxis],yf_est[:,np.newaxis])))
    f_est.set_array(sf)    
    return h_real, f_real, h_est, f_est


def plot(forest):
        """ Plots the current state """
        xh,yh,xf,yf,xb,yb = forest.grid_to_array()
        plt.scatter(xh,yh,color='g',s=100)
        plt.scatter(xf,yf,color='r',s=100)
        plt.scatter(xb,yb,color='k',s=100)
        plt.show()


if __name__ == '__main__':
    W = 5 # width
    H = 5 # height
    T = 20 # number of time steps

    # Initialize forest
    f = ff.forest(W, H)

    # Run simulation
    N = 10 # number of simulations to run
    errors = np.zeros((N,2,T))
    times = np.zeros((N,T-1))
    for i in range(N):
        errori, timei = simulate(T,f)
        errors[i,:] = errori
        times[i,:] = timei

    # Save data
    np.save('errors', errors)
    np.save('times', times)


