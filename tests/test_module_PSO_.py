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


print("test npea.PSO_.PSO")
gbestSol, gbestFit, convergence = npea.PSO_.PSO(myObjectivefunction, 
           popsize=100, w =0.5, c1 = 1.5, c2 = 1.5, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


print("test npea.PSO_.PSO")
gbestSol, gbestFit, convergence = npea.PSO_.PSO(myObjectivefunction, 
           popsize=100, w =0.5, c1 = 1.5, c2 = 1.8, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )


print("test npea.PSO_.DIPSO")
gbestSol, gbestFit, convergence = npea.PSO_.DIPSO(myObjectivefunction, 
           popsize=100, c1 = 1.5, c2 = 1.8, 
           lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)   
print( gbestFit )



input("OK")

