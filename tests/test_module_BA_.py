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


print("test npea.BA_.BA")
gbestSol, gbestFit, convergence = npea.BA_.BA(myObjectivefunction, 
           popsize=100, nb = 20, nrb = 5, ne = 5, nre = 20, ngh = 0.01, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


input("OK")


