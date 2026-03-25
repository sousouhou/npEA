"""
The implementation of different Tabu Search algorithms.

List of algorithms
------------------
    TS      Original Tabu Search    

"""
import numpy as np

  
def TS(func, nbsize:int = 30, stepscale = 0.01, tabusize:int = 50,  distol = 0.01,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Tabu Search (TS). 
    
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    nbsize : int
        Neighborhoods size.
    stepscale : float, (0,1]
        Step scale.
    tabusize : int
        Number of tabus in tabulist.
    distol : float
        Normalized Euclidean distance, determine whether two solution is close.
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
    Glover, F. (1989). Tabu search—part I. ORSA Journal on computing, 1(3), 190-206.
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    currX = rng.uniform(lb, ub, (dim))    # current X,  1D vector
    currFit = func(currX)                 # a scalar
    FEs += 1

    gbestSol = currX.copy()  # gbest solution 
    gbestFit = currFit       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] )    
    
    tabulist = np.zeros( (tabusize, dim) )  # 2D matrix
    tabulist[0] = currX
    counter = 1
    
    while FEs <= MaxFEs: 
    
        # --- generate neighborhoods ---
        nbsX = currX + stepscale * (ub-lb) * rng.uniform(-1, 1, (nbsize, dim))
        
        nbsX = np.clip(nbsX, lb, ub)  # boundary control
        nbsFit = np.apply_along_axis(func, 1, nbsX) 
        FEs += nbsize  
        
        # --- update gbest ---  
        bestId = np.argmin(nbsFit)  # the best one of neighborhoods
        if nbsFit[bestId] < gbestFit :
            gbestSol = nbsX[bestId].copy()
            gbestFit = nbsFit[bestId]

        # --- select the best one from the candidate solutions that are not in tabulist ---
        numtabu = min(counter, tabusize)  # number solutions in tabulist
        
        currtabulist = tabulist[0:numtabu] # current tabulist
        diff = currtabulist.reshape((1,numtabu, dim)) - nbsX.reshape( (nbsize,1,dim) )
        diff = diff/(ub-lb)            # normalized for all dims,
        Rmatrix = np.linalg.norm(diff, ord=2, axis=2) # distance matrix, shape is (nbsize, numtabu)
    
        flagclose = Rmatrix < distol      # determine whether two solution are close.
        mask1 = np.any(flagclose, axis=1) # a neighborhood is close to a solution in tabulist
        
        if np.all(mask1) == True:         # all neighborhoods are in tabulist
            pass
        else: 
            legalnbsX = nbsX[~mask1]    #  candidate solutions, that are not in tabulist
            legalnbsFit = nbsFit[~mask1]
            
            bestId = np.argmin(legalnbsFit)
            currX    = legalnbsX[bestId].copy() 
            currFit  = legalnbsFit[bestId]                
            
            # update tabulist
            tabulist[ counter%tabusize ] = currX
            counter +=1                

        if FEs%(int(MaxFEs/50)) < nbsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )   
            
    return (gbestSol, gbestFit, convergence) 
            

