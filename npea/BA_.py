"""
The implementation of different Bees Algorithms.

List of algorithms
------------------
    BA      Original Bees Algorithm     

"""
import numpy as np

  
def BA(func, popsize:int = 100, nb:int = 20, nrb:int = 5, ne:int = 5, nre:int = 20, ngh = 0.1, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Bees Algorithm (BA).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size, number of scout bees, ns.
    nb : int
        Number of best sites.
    nrb : int
        Recruited bees for remaining best sites.
    ne : int
        Number of elite sites.
    nre : int
        Recruited bees for elite sites.
    ngh : float
        Initial size of neighbourhood.
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
    Pham, D. T., & Castellani, M. (2009). The Bees Algorithm: Modelling foraging 
    behaviour to solve continuous optimization problems. Proceedings of the 
    Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering 
    Science, 223(12), 2919-2938.
    """    
    assert ( (popsize > nb) and  (nb > ne) )
    
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
    
        sortedSIds = np.argsort(popFit)   # ascending order
        popX   = popX[sortedSIds]
        popFit = popFit[sortedSIds]    
    
        # --- local search, elites --- 
        for i in range(0, ne):
            center = popX[i]
            recbeesX = center +  ngh * (ub-lb) * rng.uniform(-1, 1, (nre, dim))  # recruited bees

            recbeesX = np.clip(recbeesX, lb, ub)
            recbeesFit = np.apply_along_axis(func, 1, recbeesX)  # 1D vector
            FEs += nre 
            
            minid = np.argmin(recbeesFit)
            if recbeesFit[minid] < popFit[i] :
                popX[i] = recbeesX[minid]
                popFit[i] = recbeesFit[minid]
            
        # --- local search, best bees, excluding elites --- 
        for i in range(ne, nb):
            center = popX[i]
            recbeesX = center +  ngh * (ub-lb) * rng.uniform(-1, 1, (nrb, dim))  # recruited bees

            recbeesX = np.clip(recbeesX, lb, ub)
            recbeesFit = np.apply_along_axis(func, 1, recbeesX)  # 1D vector
            FEs += nrb 
            
            minid = np.argmin(recbeesFit)
            if recbeesFit[minid] < popFit[i] :
                popX[i] = recbeesX[minid]
                popFit[i] = recbeesFit[minid]
            
        # --- global search, the rest bees --- 
        numscouts = popsize - nb
        popX[nb:]   = rng.uniform(lb, ub, (numscouts, dim))  # 2D matrix
        popFit[nb:] = np.apply_along_axis(func, 1, popX[nb:]) # 1D vector, len(popFit)==popsize
        FEs += numscouts        
    
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < 2*popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  
            
    return (gbestSol, gbestFit, convergence) 
            

