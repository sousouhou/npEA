"""
The implementation of different Simulated Annealing algorithms.

List of algorithms
------------------
    SA      Original Simulated Annealing     

"""
import numpy as np

  
def SA(func, popsize:int = 10, T0 = 100.0, stepScale = 0.01,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Simulated Annealing (SA) with multiple particles.
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    T0 : float
        Temperature at time = 0.
    stepScale : float, [0,1]
        The step scale of random movement.
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
    Van Laarhoven, P. J., & Aarts, E. H. (1987). Simulated annealing. In
    Simulated annealing: Theory and applications (pp. 7-15). Dordrecht: 
    Springer Netherlands.
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
    
        T = T0 * (1.0 - FEs/MaxFEs) + 1e-15  # linearly decrease
    
        newpopX = popX + stepScale * (ub-lb) * rng.normal(0, 1, size=(popsize,dim)) # Gaussian distribution, mu = 1, sigma = 1.  
    
        newpopX = np.clip(newpopX, lb, ub)  # boundary control
        newpopFit = np.apply_along_axis(func, 1, newpopX) # 1D vector 
        FEs += popsize    
    
        delta = newpopFit - popFit
       
        accept = np.zeros(popsize, dtype=bool) # False
        mask = (delta >= 0 )
        accept[~mask] = True   # delta < 0
        accept[mask] = (rng.uniform(0, 1, np.sum(mask)) < np.exp(-delta[mask] / T) )

        popX[accept] = newpopX[accept]
        popFit[accept] = newpopFit[accept]
    
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )  
            
    return (gbestSol, gbestFit, convergence) 
            

