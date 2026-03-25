"""
The implementation of different competitive swarm optimizer algorithms.  

List of algorithms
------------------
    CSO      Original competitive swarm optimizer     

"""
import numpy as np

  
def CSO(func, popsize:int = 100, phi = 0.01, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original competitive swarm optimizer (CSO).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    phi : float
        A parameter controlling the influence of mean(popX).
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
    Cheng, R., & Jin, Y. (2014). A competitive swarm optimizer for large scale 
    optimization. IEEE transactions on cybernetics, 45(2), 191-204.
    """
    assert ( int(popsize%2) == 0  )
    
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]
    
    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popVel = np.zeros( (popsize, dim) )
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize

    bestId  = np.argmin(popFit)    
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector
    gbestFit = popFit[bestId]       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] ) 
    
    while FEs <= MaxFEs: 
    
        indexlist = np.arange(0, popsize)
        randInds = rng.permuted(indexlist)  # generate a random index list for pairwise competition 
        popX = popX[randInds]
        popVel = popVel[randInds]
        popFit = popFit[randInds]
        
        centerX = popX.mean(axis=0)  # 1D
    
        # --- pairwise competition ---
        half01X = popX[0:int(popsize/2)]
        half02X = popX[int(popsize/2): ]
        half01Fit = popFit[0:int(popsize/2)]
        half02Fit = popFit[int(popsize/2): ]
        half01Vel = popVel[0:int(popsize/2)]
        half02Vel = popVel[int(popsize/2): ]
        
        maskWin = half01Fit < half02Fit
        winX   = np.where( maskWin.reshape( (int(popsize/2),1) ), half01X, half02X ) 
        winVel = np.where( maskWin.reshape( (int(popsize/2),1) ), half01Vel, half02Vel ) 
        winFit = np.where( maskWin,                               half01Fit, half02Fit )

        maskLose = ~maskWin
        loseX   = np.where( maskLose.reshape( (int(popsize/2),1) ), half01X, half02X ) 
        loseVel = np.where( maskLose.reshape( (int(popsize/2),1) ), half01Vel, half02Vel ) 
        loseFit = np.where( maskLose,                               half01Fit, half02Fit )
        
        # --- position and velocity update of losers ---
        c1 = rng.uniform(0, 1, (int(popsize/2), dim))
        c2 = rng.uniform(0, 1, (int(popsize/2), dim))
        c3 = rng.uniform(0, 1, (int(popsize/2), dim))

        loseVel = c1 * loseVel + c2 *( winX - loseX) + phi * c3 *( centerX - loseX )
        loseX += loseVel

        # --- evaluation --
        loseX = np.clip(loseX, lb, ub)  # boundary control
        loseFit = np.apply_along_axis(func, 1, loseX)  
        FEs += int(popsize/2)    
    
        # --- merge ---
        popX = np.vstack( (winX, loseX) )
        popVel = np.vstack( (winVel, loseVel) )
        popFit = np.hstack( (winFit, loseFit) )

        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit  :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*0.5 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) ) 
            
    return (gbestSol, gbestFit, convergence) 
            

