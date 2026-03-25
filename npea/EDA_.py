"""
The implementation of different estimation of distribution algorithms.

List of algorithms
------------------
    EDA      Original estimation of distribution algorithm     

"""
import numpy as np

  
def EDA(func, popsize:int = 100, elitesize:int = 30, minstd = 1e-5,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original estimation of distribution algorithm (EDA).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    elitesize : int
        Number of elites.
    minstd : float, (0,1]
        Minimum standard deviation.
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
    Larranaga, P., & Lozano, J. A. (Eds.). (2001). Estimation of distribution 
    algorithms: A new tool for evolutionary computation (Vol. 2). Springer 
    Science & Business Media.
    """
    assert minstd > 0
    
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
    
    sortedIds = np.argsort(popFit)  # ascending order
    popX   = popX[sortedIds]
    popFit = popFit[sortedIds]  
    
    while FEs <= MaxFEs: 
    
        # --- model estimation (Gaussian) ---
        elitesX = popX[0:elitesize]
        
        elmean = elitesX.mean(axis=0)
        elstd  = elitesX.std(axis=0)        
        elstd = np.maximum(elstd, minstd)
            
        # --- sampling ---
        newpopX = rng.normal(elmean, elstd, size=(popsize, dim))  # Gaussian distribution 
        
        newpopX = np.clip(newpopX, lb, ub)  # boundary control
        newpopFit = np.apply_along_axis(func, 1, newpopX) # 1D vector, len(popFit)==popsize
        FEs += popsize    
    
        # --- merge and compete ---
        popX = np.vstack( (popX, newpopX) )
        popFit = np.hstack( (popFit, newpopFit) )
    
        sortedIds = np.argsort(popFit)  # ascending order
        popX   = popX[sortedIds]
        popFit = popFit[sortedIds]      
    
        popX   = popX[0:popsize]
        popFit = popFit[0:popsize]          
    
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  
            
    return (gbestSol, gbestFit, convergence) 
            

