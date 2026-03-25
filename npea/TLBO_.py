"""
The implementation of different Teaching–learning-based optimization algorithms.

List of algorithms
------------------
    TLBO      Original Teaching–learning-based optimization     

"""
import numpy as np

  
def TLBO(func, popsize:int = 100, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Teaching–learning-based optimization (TLBO).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
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
    Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching–learning-
    based optimization: a novel method for constrained mechanical design 
    optimization problems. Computer-aided design, 43(3), 303-315.
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
    
        # --- Teacher Phase ---
        meanSol = np.mean(popX, axis=0)  # 1D vector, shape is (dim,)
        
        TF = rng.integers(1, 3, (popsize, 1))  # 1 or 2
    
        newpopX = popX + rng.uniform(0, 1, (popsize, dim)) * ( gbestSol - TF* meanSol )
        
        newpopX = np.clip(newpopX, lb, ub)    # boundary control
        newpopFit = np.apply_along_axis(func, 1, newpopX) 
        FEs += popsize   
        
        mask = newpopFit < popFit
        popX[mask]   = newpopX[mask]
        popFit[mask] = newpopFit[mask]
        
        # --- Learner Phase ---
        # each Xi select a pair, their indexs are not equal
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame =  ( pairId==np.arange(0, popsize) )
        pairId[masksame] = (pairId[masksame] +1) % popsize          
        
        bettermask = popFit[pairId] < popFit
        temp1 = popX[pairId] - popX
        diff = np.where( bettermask.reshape((popsize,1)), temp1, -temp1)
        newpopX = popX + rng.uniform(0, 1, (popsize, dim)) * diff
        
        newpopX = np.clip(newpopX, lb, ub)    # boundary control
        newpopFit = np.apply_along_axis(func, 1, newpopX) 
        FEs += popsize   
        
        mask = newpopFit < popFit
        popX[mask]   = newpopX[mask]
        popFit[mask] = newpopFit[mask]        
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*2 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence) 
            

