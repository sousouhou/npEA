
import numpy as np

# x is 1D array (np.ndarray object)

def sphere(x):   
    return np.sum(x ** 2)

def sphere_shifted(x):   
    return np.sum( (x-9.16) ** 2)




# some simple functions in CEC 2017
def bent_cigar(x):  
    return x[0]**2 + 10**6 * np.sum(x[1:] ** 2)


def rastrigin(x):    
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
    
    
def discus(x): 
    return 10**6 * x[0]**2 + np.sum( x[1:]**2 )










