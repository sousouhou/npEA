"""
The implementation of different Evolution Strategy algorithms. 

List of algorithms
------------------
    ES      original Evolution Strategy    

"""
import numpy as np
  
  
def ES(func, popsize:int = 100, lamuda:int = 80, sigma = 0.1, adaption:bool = True, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Evolution Strategy (ES).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size, is mu in "mu + lamuda strategy".
    lamuda : int
        Kids size, "mu + lamuda strategy".
    sigma : float
        Step size.
    adaption : bool
        True, sigma is adapted.
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
    Back, Thomas, and Hans-Paul Schwefel. "An overview of evolutionary algorithms for 
    parameter optimization." Evolutionary computation 1.1 (1993): 1-23.
    https://cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html    
    """
    assert lamuda<= popsize
    
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

    popSigma = np.full((popsize, dim), sigma)  # each individual has its own sigma
    tau = 1 / np.sqrt(2 * np.sqrt(dim))        # for global noise 
    taup = 1 / np.sqrt(2 * dim)                # for local_noise

    while FEs <= MaxFEs: 
    
        # --- mutation ---
        # Create lamuda kids, by randomly selecting parent indices
        ids = rng.choice( np.arange(0,popsize), lamuda, replace=False)  
        parentsX = popX[ids]
        parentsSigma = popSigma[ids]
        
        if adaption == True:
            a = rng.normal(0, 1, size=(lamuda, dim)) 
            b = rng.normal(0, 1, size=(lamuda, dim)) 
            kidsSigma = parentsSigma * np.exp(taup * a + tau * b)
        else:
            kidsSigma = np.full((lamuda, dim), sigma)
        
        dimrange = ub-lb
        kidsX = parentsX + kidsSigma * dimrange * rng.normal(0, 1, size=(lamuda, dim)) 
        kidsX = np.clip(kidsX, lb, ub)  # boundary control
        
        kidsFit = np.apply_along_axis(func, 1, kidsX) 
        FEs += lamuda    
    
        # --- Selection  "mu + lamuda strategy" --- 
        allX = np.vstack( [popX, kidsX])
        allFit = np.hstack( [popFit, kidsFit] )
        allSigma = np.vstack( [popSigma, kidsSigma] )

        sortedIds = np.argsort(allFit)                   # ascending order
        popX     = allX[sortedIds][0:popsize]   
        popFit   = allFit[sortedIds][0:popsize] 
        popSigma = allSigma[sortedIds][0:popsize] 

        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d, gbestFit: %.18e"%(FEs, gbestFit) )  
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )   
            
    return (gbestSol, gbestFit, convergence) 
            

