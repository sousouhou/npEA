
import sys
sys.path.append('..')

import numpy as np
import npea

def myObjectivefunction(solution):   # minimization
    ndim = solution.shape[0]         # if necessary
    return np.sum( (solution-9.16)**2)

gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction, 
               popsize=100, F=0.5, Cr=0.9, 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2) 
    
print('gbestSol: ' + str(gbestSol) )
print('gbestFit: ' + str(gbestFit) )
print('convergence: \n' + str(convergence) )

input("OK")