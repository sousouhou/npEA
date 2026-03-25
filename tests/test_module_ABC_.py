import sys
sys.path.append('..')

import numpy as np
import time
import npea


import benchmarkFunctions 
# myObjectivefunction = benchmarkFunctions.sphere    
# myObjectivefunction = benchmarkFunctions.sphere_shifted    
# myObjectivefunction = benchmarkFunctions.bent_cigar    
# myObjectivefunction = benchmarkFunctions.rastrigin    
myObjectivefunction = benchmarkFunctions.discus    


print("test npea.ABC_.ABC")
gbestSol, gbestFit, convergence = npea.ABC_.ABC(myObjectivefunction, 
           popsize=100, limit=200, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


print("test npea.ABC_.GABC")
gbestSol, gbestFit, convergence = npea.ABC_.GABC(myObjectivefunction, 
           popsize=100, limit=200, C=1.5,
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )

print("test npea.ABC_.MABC")
gbestSol, gbestFit, convergence = npea.ABC_.MABC(myObjectivefunction, 
           popsize=100, limit=200, MR = 0.4, SF =1.5,
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


input("OK")

