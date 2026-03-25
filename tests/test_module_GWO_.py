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


print("test npea.GWO_.GWO, 1st")
gbestSol, gbestFit1, convergence = npea.GWO_.GWO(myObjectivefunction, 
           popsize=100, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit1 )


print("test npea.GWO_.GWO, 2nd")
gbestSol, gbestFit2, convergence = npea.GWO_.GWO(myObjectivefunction, 
           popsize=100, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=4)                  
print( gbestFit2 )



input("OK")

