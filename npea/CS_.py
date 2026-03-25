"""
The implementation of different Cuckoo search algorithms.

List of algorithms
------------------
    CS      Original Cuckoo search     

"""
import numpy as np
import math

  
def CS(func, popsize:int = 100, pa = 0.2, alpha = 0.05, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Cuckoo search (CS).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.  
    pa : float
        Probability of abandoned worst nests.
    alpha : float
        Step size.
    lb : list of float
        Lower bounds of each dimension, the dimension of the optimization problem
        is calculated as len(lb).
    ub : list of float
        Upper bounds of each dimension.
    MaxFEs : int
        Maximum number of function evaluations.
    seed : int
        A seed to initialize random number generator.

    Returns
    -------
    gbestSol : 1D ndarray
        Global best solution found by this algorithm.
    gbestFit: float
        Objective function value of gbestSol.
    convergence : 2D ndarray
        Convergence history, shape is (~50, 2), the 1st column is FEs, the 2nd
        column is gbestFit.
        
    References
    ----------
    Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." 2009 World
    congress on nature & biologically inspired computing (NaBIC). Ieee, 2009.
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize

    bestId  = np.argmin(popFit)    
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector
    gbestFit = popFit[bestId]       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] )    
    
    beta = 1.5 
    usigma = math.pow(math.gamma(1. + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2.) * beta * math.pow(2., (beta - 1) / 2)), 1. / beta)

    while FEs <= MaxFEs: 
    
        # --- levy_flight --- 
        u = rng.normal(0, usigma, size=(popsize, dim))
        v = rng.normal(0, 1     , size=(popsize, dim))
        step = u / np.power(np.abs(v), 1.0/beta)
                                          # Here paper not state
        newpopX = popX + alpha * step * ( popX - gbestSol )
        
        newpopX = np.clip(newpopX, lb, ub)  # boundary control
        newpopFit = np.apply_along_axis(func, 1, newpopX) 
        FEs += popsize    
    
        improved = ( newpopFit < popFit )
        popX[improved]   = newpopX[improved]
        popFit[improved] = newpopFit[improved]       
         
        # --- abandoned worst nests ---
        sortedSIds = np.argsort(popFit)   # ascending order
        popX   = popX[sortedSIds]
        popFit = popFit[sortedSIds]
        
        abanSize = int( popsize*pa )      # abandoned size
        
        # random generate new solutions , Here paper not state. 
        popsomeX = rng.uniform(lb, ub, (abanSize, dim))   
        popsomeFit = np.apply_along_axis(func, 1, popsomeX)
        FEs += abanSize        
 
        popX[-abanSize: ] = popsomeX[:]
        popFit[-abanSize: ] = popsomeFit[:]

        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*2 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  
            
    return (gbestSol, gbestFit, convergence) 
            

