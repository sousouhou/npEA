
import sys
sys.path.append('..')

import numpy as np
import npea
import matplotlib.pyplot as plt

def myObjectivefunction(solution):   # minimization
    return np.sum( (solution-9.16)**2)


_ , _ , DEconvergence = npea.DE_.DE(myObjectivefunction, 
               popsize=100, F = 0.5, Cr = 0.9, 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=50000, seed=2) 
    
_ , _ , PSOconvergence = npea.PSO_.PSO(myObjectivefunction, 
               popsize=100, w = 0.5, c1 = 1.5, c2 = 1.5 , 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=50000, seed=2) 

_ , _ , GAconvergence = npea.GA_.GA(myObjectivefunction, 
               popsize=100, k = 3 , pc = 0.9, pm = 0.1, 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=50000, seed=2) 


fig, ax = plt.subplots()

ax.plot(  DEconvergence[:,0], DEconvergence[:,1],  label = "DE")
ax.plot( PSOconvergence[:,0], PSOconvergence[:,1], label = "PSO")
ax.plot(  GAconvergence[:,0], GAconvergence[:,1],  label = "GA")

ax.set_xlabel('FEs',fontsize=12) 
ax.set_ylabel('gbestFit',fontsize=12) 
plt.yscale('log')  # y tick uses log
plt.legend()
plt.show()

input("OK")