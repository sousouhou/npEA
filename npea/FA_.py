"""
The implementation of different Firefly Algorithms.

List of algorithms
------------------
    FA      Original Firefly Algorithm     

"""
import numpy as np

  
def FA(func, popsize:int = 100, alpha = 0.001, beta0 = 1.0, gamma = 1.0,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Firefly Algorithm (FA).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    alpha : float, [0, 1]
        Randomization parameter.
    beta0 : float
        Attractiveness at r = 0.
    gamma : float
        Light absorption coefficient. 
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
    Yang, X. S. (2010). Firefly algorithm, stochastic test functions and design 
    optimisation. International journal of bio-inspired computation, 2(2), 78-84.
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
    
    sortedIds = np.argsort(popFit)  # ascending order
    popX   = popX[sortedIds]
    popFit = popFit[sortedIds]
    
    while FEs <= MaxFEs: 
         # Comments: FA can not be implemented by using matrix for parallel computing.
         #           This is because the ith individuals learns from its foregoing j individuals, and be successively evaluated.
         #           It leads to a decrease in computing speed.
        for i in range(0, popsize):  
            for j in range(0, i-1):  # must use two for-loops
            
                if popFit[j] < popFit[i] :  # j is better than i
                
                    norDiff = (popX[j] - popX[i]) / ( ub - lb)   # normalized diff, based on my understanding
                    r2 = np.sum( norDiff**2 )
                    beta = (beta0) * np.exp( -1.0* gamma * r2) 
                    
                    # Random movement
                    ramo = alpha * ( rng.normal(0,1, (dim)) ) * ( ub - lb )
                    
                    # Move toward brighter firefly
                    popX[i] = popX[i] + beta * (popX[j] - popX[i]) + ramo
    
                    popX[i] = np.clip(popX[i], lb, ub)
                    
                    popFit[i] = func( popX[i] )
                    FEs += 1
                    
        sortedIds = np.argsort(popFit)  # ascending order
        popX   = popX[sortedIds]
        popFit = popFit[sortedIds]
     
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
            
        if FEs%(int(MaxFEs/50)) < popsize*(popsize-1)/4 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
        
    return (gbestSol, gbestFit, convergence) 
            

