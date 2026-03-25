"""
The implementation of different Biogeography-Based Optimization algorithms.

List of algorithms
------------------
    BBO      original Biogeography-Based Optimization     

"""
import numpy as np

  
def BBO(func, popsize:int = 100, numElites:int = 2, pMutate = 0.02, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Biogeography-Based Optimization (BBO).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    numElites : int
        Number of Elites.
    pMutate : float
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
    Simon, Dan. "Biogeography-based optimization." IEEE transactions on 
    evolutionary computation 12.6 (2008): 702-713.        
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
    
    mu = (popsize - np.arange(0, popsize))/(popsize+1)     # emigration rate
    lamb = 1 - mu                                 # lambda,  immigration rate 
    
    while FEs <= MaxFEs: 
    
        # --- Save elites ---
        sortedIds = np.argsort(popFit)  # ascending order
        popX   = popX[sortedIds]
        popFit = popFit[sortedIds]
    
        elitesX   = popX[0:numElites].copy()
        elitesFit = popFit[0:numElites].copy() 
    
        # --- Emigrate, roulette wheel selection ---
        probmu = mu/mu.sum()
                                                # replace is True, meaning that a value can be selected multiple times.
        dornorsId = rng.choice(np.arange(0,popsize), size=popsize, replace=True, p=probmu )
        dornorsX   = popX[dornorsId]
        dornorsFit = popFit[dornorsId]
    
        mask1 = rng.uniform(0,1, (popsize, dim) ) < ( lamb.reshape(popsize,1) * np.ones((popsize,dim)) )
        Z = np.where(mask1, dornorsX, popX)
        
        # --- Mutation ---        
        randX = rng.uniform(lb, ub, (popsize, dim)) 
        mask2 = rng.uniform(0,1, (popsize, dim) ) < pMutate
        Z = np.where(mask2, randX, Z)
        
        # --- Evaluate ---
        popX   = np.clip(Z, lb, ub)  # boundary control, it seems unnecessary.
        popFit = np.apply_along_axis(func, 1, popX) 
        FEs += popsize        

        # --- Replace the worst solutions with elites ---
        sortedIds = np.argsort(popFit)  # ascending order
        popX   = popX[sortedIds]
        popFit = popFit[sortedIds]    
    
        popX[-numElites: ]   = elitesX[:]
        popFit[-numElites: ] = elitesFit[:]
    
        # --- Update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) ) 
            
    return (gbestSol, gbestFit, convergence) 
            




