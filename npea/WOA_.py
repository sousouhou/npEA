"""
The implementation of different Whale Optimization Algorithms.

List of algorithms
------------------
    WOA      Original Whale Optimization Algorithm     

"""
import numpy as np


def WOA(func, popsize:int = 100, b = 1.0, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Whale Optimization Algorithm (WOA).
       
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    b : flaot
        A constant for defining the shape of the logarithmic spira. 
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
    Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. 
    Advances in engineering software, 95, 51-67.  
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector
    FEs += popsize

    bestId  = np.argmin(popFit)    
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector
    gbestFit = popFit[bestId]       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] )  
    
    while FEs <= MaxFEs: 
        
        a = 2 - 2 * FEs / MaxFEs
        r1 = rng.uniform(0,1, (popsize, dim) )
        r2 = rng.uniform(0,1, (popsize, dim) )
        
        A = 2 * a * r1 - a   # shape is (popsize, dim)
        C = 2 * r2           # shape is (popsize, dim)
    
        l = rng.uniform(-1, 1, (popsize, 1))  
        p = rng.uniform( 0, 1, (popsize, 1) ) 
        
        #---- if2 in article ----
        randIds = rng.integers(0, popsize, popsize)   # 1D vector 
        Drand = np.abs( C * popX[randIds] - popX )    # shape is (popsize, dim)
        newpopX1 = popX[randIds] - A * Drand          # shape is (popsize, dim)
    
        Dbest = np.abs( C * gbestSol - popX)          # shape is (popsize, dim)
        newpopX2 = gbestSol - A * Dbest               # shape is (popsize, dim)
    
        temppopX = np.where( np.abs(A)>=1, newpopX1, newpopX2)
    
        #---- if1 in article, Spiral updating ----
                    # SpiralpopX shape is (popsize, dim)
        SpiralpopX = np.abs(gbestSol - popX) * np.exp(b * l) * np.cos(2 * np.pi * l) + gbestSol  

        popX = np.where( p<0.5, temppopX, SpiralpopX)
        
        # --- evaluate ---
        popX = np.clip(popX, lb, ub)   # boundary control
        popFit = np.apply_along_axis(func, 1, popX) 
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
            

