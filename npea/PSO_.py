"""
The implementation of different particle swarm optimization algorithms.

List of algorithms
------------------
    PSO      Standard particle swarm optimization     
    DIPSO    Decreasing Inertia Particle Swarm Optimization
"""
import numpy as np


def PSO(func, popsize:int = 100, w = 0.5, c1 = 1.5, c2 = 1.5 , 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The standard particle swarm optimization (PSO).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    w : float
        Inertia weight.
    c1 : float
        Cognitive weight.
    c2 : float
        Social weight. 
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
    Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. 
    In Proceedings of ICNN'95-international conference on neural networks (Vol. 4, 
    pp. 1942-1948). ieee.  
    Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer." 
    Evolutionary computation proceedings. Vol. 890. 1998.
    
    """
    print(locals())
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    popX   = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize
    popV   = np.zeros( (popsize, dim) )
    
    pbestX   = popX.copy()
    pbestFit = popFit.copy()
    
    bestId  = np.argmin(pbestFit)         
    gbestSol = pbestX[bestId].copy()     # gbest solution, 1D vector
    gbestFit = pbestFit[bestId]          # gbest fitness    
    convergence = np.array( [[FEs, gbestFit]] )
    
    while FEs <= MaxFEs: 
    
        r1 = rng.uniform(0,1, (popsize, dim) )
        r2 = rng.uniform(0,1, (popsize, dim) )
        
        # --- update velocity ---
        popV = w * popV   +  c1 * r1 * (pbestX - popX) + c2 * r2 * (gbestSol - popX)
        # --- update position ---
        popX = popX + popV
        popX = np.clip(popX, lb, ub)  # boundary control
        
        # --- evaluate ---
        popFit = np.apply_along_axis(func, 1, popX) 
        FEs += popsize      
        
        # --- update pbest ---
        mask =  popFit < pbestFit
        pbestX   = np.where(mask.reshape((popsize,1)), popX, pbestX)
        pbestFit = np.where(mask, popFit, pbestFit)        
        
        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence)
            


def DIPSO(func, popsize:int = 100, c1 = 1.5, c2 = 1.5 , 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """Decreasing Inertia Particle Swarm Optimization (DIPSO).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    c1 : float
        Cognitive weight.
    c2 : float
        Social weight. 
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
    Shi, Yuhui, and Russell C. Eberhart. "Parameter selection in particle swarm 
    optimization." International conference on evolutionary programming. Berlin, 
    Heidelberg: Springer Berlin Heidelberg, 1998.
    """
    print(locals())
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    popX   = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize
    popV   = np.zeros( (popsize, dim) )
    
    pbestX   = popX.copy()
    pbestFit = popFit.copy()
    
    bestId  = np.argmin(pbestFit)         
    gbestSol = pbestX[bestId].copy()     # gbest solution, 1D vector
    gbestFit = pbestFit[bestId]          # gbest fitness    
    convergence = np.array( [[FEs, gbestFit]] )
    
    while FEs <= MaxFEs: 
    
        r1 = rng.uniform(0,1, (popsize, dim) )
        r2 = rng.uniform(0,1, (popsize, dim) )
        
        w = 0.9 - (FEs*1.0/MaxFEs)*(0.9-0.4)  # modified
        # --- update velocity ---
        popV = w * popV   +  c1 * r1 * (pbestX - popX) + c2 * r2 * (gbestSol - popX)
        # --- update position ---
        popX = popX + popV
        popX = np.clip(popX, lb, ub)  # boundary control
        
        # --- evaluate ---
        popFit = np.apply_along_axis(func, 1, popX) 
        FEs += popsize      
        
        # --- update pbest ---
        mask =  popFit < pbestFit
        pbestX   = np.where(mask.reshape((popsize,1)), popX, pbestX)
        pbestFit = np.where(mask, popFit, pbestFit)        
        
        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence)
            
