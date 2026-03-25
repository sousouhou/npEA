import sys
sys.path.append('..')

import numpy as np
import time
import npea


import benchmarkFunctions 
# myObjectivefunction = benchmarkFunctions.sphere    
myObjectivefunction = benchmarkFunctions.sphere_shifted    
# myObjectivefunction = benchmarkFunctions.bent_cigar    
# myObjectivefunction = benchmarkFunctions.rastrigin    
# myObjectivefunction = benchmarkFunctions.discus   

def myObjectivefunction2(solution):   # minimization
    return np.sum( (solution)**2)



print("test npea.FWA_.FWA, 1st")
gbestSol, gbestFit, convergence = npea.FWA_.FWA(myObjectivefunction, 
           popsize=5, m = 50, mhat = 5, a= 0.04, b = 0.8, Ahat = 0.1,
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=4)   
print( gbestFit )


print("test npea.FWA_.FWA, 2nd")
gbestSol, gbestFit, convergence = npea.FWA_.FWA(myObjectivefunction2, 
           popsize=5, m = 50, mhat = 5, a= 0.04, b = 0.8, Ahat = 0.1,
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=4)   
print( gbestFit )



input("OK")

