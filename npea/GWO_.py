"""
The implementation of different Grey Wolf Optimizer algorithms.

List of algorithms
------------------
    GWO      Original Grey Wolf Optimizer     

"""
import numpy as np

  
def GWO(func, popsize:int = 100, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Grey Wolf Optimizer (GWO).
        
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
    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. 
    Advances in engineering software, 69, 46-61.
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    popX = rng.uniform(lb, ub, (popsize, dim))  
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize

    bestId  = np.argmin(popFit)    
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector
    gbestFit = popFit[bestId]       # gbest fitness 
    convergence = np.array( [[FEs, gbestFit]] )  
    
    sortedIds = np.argsort(popFit)
    popX   = popX[sortedIds]
    popFit = popFit[sortedIds]
    
    alphax = popX[0].copy()   # 1D vector
    betax  = popX[1].copy() 
    deltax = popX[2].copy() 
    
    alphafit = popFit[0]
    betafit  = popFit[1]
    deltafit = popFit[2]
   
    while FEs <= MaxFEs: 
        # --- Update the Position of search agents ---
        a = 2 - 2 * (FEs*1.0 / MaxFEs)   # linearly decreasing from 2 to 0
    
        r1 = rng.uniform(0,1, (popsize, dim) )
        r2 = rng.uniform(0,1, (popsize, dim) )
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1 = rng.uniform(0,1, (popsize, dim) )
        r2 = rng.uniform(0,1, (popsize, dim) )
        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1 = rng.uniform(0,1, (popsize, dim) )
        r2 = rng.uniform(0,1, (popsize, dim) )
        A3 = 2 * a * r1 - a
        C3 = 2 * r2    
        
        X1 = alphax - A1 * np.abs(C1 * alphax - popX)
        X2 = betax  - A2 * np.abs(C2 * betax  - popX)
        X3 = deltax - A3 * np.abs(C3 * deltax - popX)      
        
        popX = (X1 + X2 + X3) / 3.0
        popX = np.clip(popX, lb, ub)
        popFit = np.apply_along_axis(func, 1, popX)
        FEs += popsize
        
        # --- update alpha beta delta  ---
        sortedIds = np.argsort(popFit)
        popX   = popX[sortedIds]
        popFit = popFit[sortedIds]
        
        mask = popFit < alphafit   # may several ones are better than alphafit
        if np.any(mask) == True :
            alphax   = popX[mask][0].copy()   
            alphafit = popFit[mask][0]

        mask = ( alphafit < popFit) * (popFit < betafit) # * is AND . May several ones are better 
        if np.any(mask) == True :
            betax   = popX[mask][0].copy()   
            betafit = popFit[mask][0]

        mask = ( betafit < popFit) * (popFit < deltafit)  # * is AND . May several ones are better 
        if np.any(mask) == True :
            deltax   = popX[mask][0].copy()   
            deltafit = popFit[mask][0]
            
        # --- update gbest ---   
        if alphafit < gbestFit :
            gbestSol = alphax.copy()
            gbestFit = alphafit
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  
            
    return (gbestSol, gbestFit, convergence) 
            

