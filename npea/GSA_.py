"""
The implementation of different Gravitational Search Algorithms. 

List of algorithms
------------------
    GSA      Original Gravitational Search Algorithm     

"""
import numpy as np


def GSA(func, popsize:int = 100, G0 = 100.0, alpha = 20.0, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Gravitational Search Algorithm (GSA).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    G0 : float
        Initial gravitational constant.
    alpha : float
        A parameter to calculate G.
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
    Rashedi, E., Nezamabadi-Pour, H., & Saryazdi, S. (2009). GSA: a gravitational 
    search algorithm. Information sciences, 179(13), 2232-2248.
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
    popV = np.zeros((popsize, dim))  # 2D matrix

    bestId  = np.argmin(popFit)    
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector
    gbestFit = popFit[bestId]       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] )  
    
    while FEs <= MaxFEs: 
     
        # Mass calculation
        worstfit = np.max(popFit)
        bestfit  = np.min(popFit)
        
        popM = np.ones(popsize) / popsize  # Mass of each individual, 1D vector
        if np.fabs(worstfit - bestfit) > 1e-15 : # not equal
            q = (worstfit - popFit) / ( worstfit - bestfit ) # 1D
            popM = q / np.sum(q)
        
        # update G
        G = G0 * np.exp(-alpha * FEs*1.0 / MaxFEs)
    
        # select K best 
        numKbest = popsize * int(1 - FEs*1.0 / MaxFEs)
        numKbest = max(1, numKbest)  # numKbest >=1
    
        sortedIds = np.argsort(popM)   # ascending
        sortedIds = sortedIds[::-1] # descending
        selIds = sortedIds[:numKbest]  # selected Ids
        KbestsX = popX[selIds]
        KbestsM = popM[selIds]
    
                # diff is (popsize, numKbest , dim), diff is (popj - popi)
        diff = np.reshape(KbestsX, (1, numKbest, dim) ) - np.reshape(popX, (popsize, 1 , dim) )  
        Rmatrix = np.linalg.norm(diff, ord=2, axis=2)  # Rmatrix is （popsize, numKbest), 
                                 
        # ensure i is not j. 
        ind1 = np.ones((popsize,numKbest), dtype=int) * selIds
        ind2 = np.ones((popsize,numKbest), dtype=int) * np.arange(0, popsize).reshape((popsize,1))
        EqualMatrix =  ( ind1 == ind2 )
               
        temp1 = G * np.reshape(popM, (popsize,1) ) * np.reshape(KbestsM, (1, numKbest) ) / ( Rmatrix + 1e-15)
        # forceAm is （popsize, numKbest)
        forceAm = np.where(EqualMatrix, 0, temp1)
        
        rng01 = rng.uniform(0,1, (popsize, numKbest, dim) )
        force = np.sum( rng01 * np.reshape(forceAm, (popsize, numKbest, 1) ) * diff, axis=1)  # (popsize , dim)
        
        acc =  force / ( np.reshape(popM, (popsize,1)) + 1e-15)  # (popsize , dim) 
        popV = rng.uniform(0,1, (popsize, dim) ) * popV + acc
        
        popX = popX + popV 
        popX = np.clip(popX, lb, ub)  # boundary control
        popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
        FEs += popsize        
    
    
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence) 
            

