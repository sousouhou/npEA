"""
The implementation of different sine cosine algorithms. 

List of algorithms
------------------
    SCA        Original sine cosine algorithm     
    SCAgreedy  Sine cosine algorithm (SCA) with greedy strategy
"""
import numpy as np

  
def SCA(func, popsize:int = 100, a = 2.0,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original sine cosine algorithm (SCA).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    a : float
        A constant for calculating r1.
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
    Mirjalili, S. (2016). SCA: a sine cosine algorithm for solving optimization 
    problems. Knowledge-based systems, 96, 120-133.
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
    
    while FEs <= MaxFEs: 
    
        r1 = a - a * (FEs / MaxFEs)          # linearly decreasing
        r2 = rng.uniform(0, 2*np.pi, (popsize, dim))    
        r3 = rng.uniform(0, 2      , (popsize, dim))    
        r4 = rng.uniform(0, 1      , (popsize, dim))    
    
        temp1 = popX + r1 * np.sin(r2) * np.abs(r3 * gbestSol - popX)
        temp2 = popX + r1 * np.cos(r2) * np.abs(r3 * gbestSol - popX)
    
        newpopX = np.where( r4<0.5, temp1, temp2)
    
        newpopX = np.clip(newpopX, lb, ub)  # boundary control,
        newpopFit = np.apply_along_axis(func, 1, newpopX) 
        FEs += popsize  
        
        popX = newpopX
        popFit = newpopFit
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit  :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
            
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  

    return [gbestSol, gbestFit, convergence]



def SCAgreedy(func, popsize:int =100, a = 2.0,
    lb :list  = None, ub :list  = None,  
    MaxFEs: int =100000, seed :int =42):
    """The sine cosine algorithm (SCA) with greedy strategy.
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    a : float
        A constant for calculating r1.
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
    Mirjalili, S. (2016). SCA: a sine cosine algorithm for solving optimization 
    problems. Knowledge-based systems, 96, 120-133.
    """
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
    
    while FEs <= MaxFEs: 
    
        r1 = a - a * (FEs / MaxFEs)          # linearly decreasing
        r2 = rng.uniform(0, 2*np.pi, (popsize, dim))    
        r3 = rng.uniform(0, 2      , (popsize, dim))    
        r4 = rng.uniform(0, 1      , (popsize, dim))    
    
        temp1 = popX + r1 * np.sin(r2) * np.abs(r3 * gbestSol - popX)
        temp2 = popX + r1 * np.cos(r2) * np.abs(r3 * gbestSol - popX)
    
        newpopX = np.where( r4<0.5, temp1, temp2)
    
        newpopX = np.clip(newpopX, lb, ub)  # boundary control,
        newpopFit = np.apply_along_axis(func, 1, newpopX) 
        FEs += popsize  
        
        # --- greedy strategy ---
        mask = newpopFit < popFit
        popX[mask] = newpopX[mask]
        popFit[mask] = newpopFit[mask]
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit  :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
            
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  

    return (gbestSol, gbestFit, convergence) 


