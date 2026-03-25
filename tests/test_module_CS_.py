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


print("test npea.CS_.CS")
gbestSol, gbestFit, convergence = npea.CS_.CS(myObjectivefunction, 
           popsize=100, pa=0.2, alpha=0.05,
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


input("OK")


