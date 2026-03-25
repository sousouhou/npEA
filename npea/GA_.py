"""
The implementation of different genetic algorithms.

List of algorithms
------------------
    GA      original genetic algorithm     

"""
import numpy as np

def GA(func, popsize:int = 100, k:int = 3 , pc = 0.9, pm = 0.1, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original genetic algorithm (GA). 

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    k : int
        k-way tournament selection.
    pc : float
        Probability of crossover.
    pm : float
        Probability of mutation.
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
    Holland, J. H. (1992). Genetic algorithms. Scientific american, 267(1), 66-73.
    https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]
    halfsize  = popsize//2

    popX   = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize

    bestId  = np.argmin(popFit)      
    gbestSol = popX[bestId].copy()     # gbest solution, 1D vector
    gbestFit = popFit[bestId]          # gbest fitness    
    convergence = np.array( [[FEs, gbestFit]] )
    
    while FEs <= MaxFEs: 

        # -- Selection --
        # tournament selection  
        tou1Id = np.ones( (halfsize, popsize), dtype=int) * np.arange(0,popsize) 
        tou1Id = rng.permuted(tou1Id, axis=1)
        tou1Id = tou1Id[:, 0:k]      # selection k for competing
        tou1F = popFit[tou1Id]
        dadsId = tou1Id[np.arange(0,halfsize), np.argmin(tou1F, axis=1)]  # 1D vector

        tou1Id = np.ones( (halfsize, popsize), dtype=int) * np.arange(0,popsize)
        tou1Id = rng.permuted(tou1Id, axis=1)
        tou1Id = tou1Id[:, 0:k]      # selection k for competing
        tou1F = popFit[tou1Id]
        momsId = tou1Id[np.arange(0,halfsize), np.argmin(tou1F, axis=1)]  # 1D vector
        
        dadsX = popX[dadsId]
        momsX = popX[momsId]

        # --- Uniform Crossover  ---
        mask = rng.uniform(0,1, (halfsize, dim) ) < 0.5   
        tempkids1 = np.where(mask, dadsX, momsX)  # crossover all
        tempkids2 = np.where(mask, momsX, dadsX)
        
        doCross = ( rng.uniform(0,1, halfsize) < pc ).reshape(halfsize, 1)
        kids1 = np.where(doCross, tempkids1, dadsX) 
        kids2 = np.where(doCross, tempkids2, momsX)

        # --- Mutation, multi points --- 
        doMutate = ( rng.uniform(0,1, (halfsize, dim) ) < pm )
        randX = rng.uniform(lb, ub, (halfsize, dim)) 
        kids1 = np.where(doMutate, randX, kids1)
        
        doMutate = ( rng.uniform(0,1, (halfsize, dim) ) < pm )
        randX = rng.uniform(lb, ub, (halfsize, dim)) 
        kids2 = np.where(doMutate, randX, kids2)
        
        # --- Survivor, kids and popX compete --- 
        kidsX = np.vstack( (kids1, kids2) )
        
        kidsX = np.clip(kidsX, lb, ub)
        kidsFit = np.apply_along_axis(func, 1, kidsX)
        FEs += popsize
        
        popX = np.vstack( (popX, kidsX) )      
        popFit = np.hstack( (popFit, kidsFit) ) 
        sortId = np.argsort( popFit )
        popX = popX[ sortId[0:popsize] ]         
        popFit = popFit[ sortId[0:popsize] ]    
        
        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]       
          
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence)
            

