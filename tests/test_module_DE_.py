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


print("test npea.DE_.DE")
gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction, 
           popsize=100, F=0.5, Cr=0.9, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


print("test npea.DE_.DEmu")
gbestSol, gbestFit, convergence = npea.DE_.DEmu(myObjectivefunction, 
           popsize=100, F=0.5, Cr=0.1, strategy = 's2',
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*50, seed=2) 
print( gbestFit )


print("test npea.DE_.jDE")
gbestSol, gbestFit, convergence = npea.DE_.jDE(myObjectivefunction, 
           popsize=100, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*50, seed=2) 
print( gbestFit )



input("OK")