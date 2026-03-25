
import sys
sys.path.append('..')

import numpy as np
import npea

import time

def myObjectivefunction(solution):   # minimization
    return np.sum( (solution-9.16)**2)


start = time.perf_counter()

gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction, 
               popsize=100, F=0.5, Cr=0.9, 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2) 

end = time.perf_counter()  

print('Running %.8f seconds'%(end-start))  
        

input("OK")
