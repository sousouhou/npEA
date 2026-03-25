# An example to show training a linear regression model. 

import sys
sys.path.append('..')

import numpy as np
import npea


# The data should be loaded outside the scope of myObjectivefunction.
trainX = np.array([[1.0, 2.0],
                   [3.0, 4.0],
                   [6.0, 6.0],
                   [7.0, 7.0]])
trainY = np.array([10.0, 11.0, 12.0, 13.0])     


def myObjectivefunction(solution):   # minimization
    w = solution[0:2]  # decode
    b = solution[2]
    # Do not load data here; otherwise, it will slow down execution.
    predictY = trainX.dot(w) + b
    return np.sum( ( trainY - predictY )**2 )  # mean square error


gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction, 
               popsize=100, F=0.5, Cr=0.9, 
               lb = [-100,]*3 , ub = [ 100,]*3 , MaxFEs=10000, seed=42) 


print('gbestSol: ' + str(gbestSol) )
print('gbestFit: ' + str(gbestFit) )
print('convergence: \n' + str(convergence) )

input("OK")