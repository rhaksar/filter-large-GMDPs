from matplotlib import pyplot as plt
import pdb
import numpy as np

times = np.load('times.npy')
errors = np.load('errors.npy')

N = errors.shape[0]
T = errors.shape[2]

avgErrors = np.zeros((errors.shape[1], T))
avgTimes = np.zeros((times.shape[0], T-1))
for i in range(errors.shape[0]):
    avgErrors += errors[i,:]
    avgTimes += times[i,:]

avgErrors = avgErrors/N
avgTimes = avgTimes/N

plt.figure()
plt.plot(range(T), avgErrors[0,:], 'r', label='LBP', linewidth=3)
plt.plot(range(T), avgErrors[1,:], 'k', label='Measurement', linewidth=3)
plt.ylim([0,16])
plt.xlabel('Time', fontsize=20)
plt.ylabel('Errors (%)', fontsize=20)
plt.title('Estimation Errors vs. Time', fontsize=20)
plt.legend(loc='best', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.figure()
plt.plot(range(T-1), avgTimes[0,:], 'r', label='LBP', linewidth=3)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Computation Time (s)', fontsize=20)
plt.title('Computation Time vs. Time', fontsize=20)
plt.legend(loc='best', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

