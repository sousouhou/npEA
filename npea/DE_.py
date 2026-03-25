"""
The implementation of different Differential Evolution algorithms.

List of algorithms
------------------
    DE      Original Differential Evolution    
    DEmu    Differential Evolution with different mutation strategies
    jDE     Self-adapting control parameters in differential evolution

"""
import numpy as np

  
def DE(func, popsize:int = 100, F = 0.5, Cr = 0.9, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Differential Evolution (DE).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    F : float
        Saling factor.
    Cr : float
        Crossover rate.
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
    Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient 
    heuristic for global optimization over continuous spaces. Journal of global 
    optimization, 11(4), 341-359.
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
        # -- Mutation --
        # DE/rand/1
        # three random indexes, they are not equal.
        a = np.ones( (popsize, popsize), dtype=int) * np.arange(0, popsize)
        idM = rng.permuted(a, axis=1)  # each slice along the given axis is shuffled independently of the others.
        idM = idM[:, 0:3]      # 1st 2nd 3rd columns are needed. 
        
        V = popX[ idM[:,0], :] + F*( popX[ idM[:,1], :] - popX[ idM[:,2], :] ) # V.shape == popX.shape
        
        # -- Crossover --
        a = rng.uniform(0,1, (popsize, dim) )
        # ensure at least one dimension is exchanged
        b = np.zeros( (popsize, dim), dtype=bool)
        b[:,0] = True
        c = rng.permuted(b, axis=1) 
        # bool matrix for Crossover
        crossM = (a<Cr) + c
        
        U = np.where(crossM, V, popX)  # U.shape == popX.shape
        U = np.clip(U, lb, ub)  # boundary control
        
        # -- Selection --
        UFit = np.apply_along_axis(func, 1, U)  # 1D vector
        FEs += popsize
        
        mask = ( UFit <= popFit )   # bool 1D vector 
        popX = np.where(mask.reshape((popsize,1)), U, popX)
        popFit = np.where(mask, UFit, popFit)
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit  :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence)




def DEmu(func, popsize:int = 100, F = 0.5, Cr = 0.9, strategy = 's0',
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The Differential Evolution with different mutation strategies.

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    F : float
        Saling factor.
    Cr : float
        Crossover rate.
    strategy : {'s0', 's1', 's2', 's3', 's4'} 
        Select different mutation strategy.
        s0 : DE/rand/1
        s1 : DE/best/1
        s2 : DE/current-to-best/1
        s3 : DE/best/2
        s4 : DE/rand/2
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
    Das, Swagatam, Sankha Subhra Mullick, and Ponnuthurai N. Suganthan. "Recent 
    advances in differential evolution–an updated survey." Swarm and evolutionary 
    computation 27 (2016): 1-30.
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
        # -- Mutation --
        a = np.ones( (popsize, popsize), dtype=int) * np.arange(0, popsize)
        idM = rng.permuted(a, axis=1)  # each slice along the given axis is shuffled independently of the others.
        idM = idM[:, 0:5]      # 1st - 5th columns are needed. 
        
        V = 0
        if strategy == 's0' : # s0 : DE/rand/1
            V = popX[ idM[:,0], :] + F*( popX[ idM[:,1], :] - popX[ idM[:,2], :] )
        if strategy == 's1' : # s1 : DE/best/1
            V = gbestSol + F*( popX[ idM[:,0], :] - popX[ idM[:,1], :] )     
        if strategy == 's2' : # s2 : DE/current-to-best/1
            V = popX + F*( gbestSol - popX )  + F*( popX[ idM[:,0], :] - popX[ idM[:,1], :] ) 
        if strategy == 's3' : # s3 : DE/best/2
            V = gbestSol + F*( popX[ idM[:,0], :] - popX[ idM[:,1], :] ) +  F*( popX[ idM[:,2], :] - popX[ idM[:,3], :] )
        if strategy == 's4' : # s4 : DE/rand/2
            V = popX[ idM[:,0], :] + F*( popX[ idM[:,1], :] - popX[ idM[:,2], :] ) + F*( popX[ idM[:,3], :] - popX[ idM[:,4], :] )  
        if strategy not in ['s0', 's1', 's2', 's3', 's4'] :
            V = popX[ idM[:,0], :] + F*( popX[ idM[:,1], :] - popX[ idM[:,2], :] ) # default is s0 : DE/rand/1
        
        # -- Crossover --
        a = rng.uniform(0,1, (popsize, dim) )
        # ensure at least one dimension is exchanged
        b = np.zeros( (popsize, dim), dtype=bool)
        b[:,0] = True
        c = rng.permuted(b, axis=1) 
        # bool matrix for Crossover
        crossM = (a<Cr) + c
        
        U = np.where(crossM, V, popX)  # U.shape == popX.shape
        U = np.clip(U, lb, ub)  # boundary control
        
        # -- Selection --
        UFit = np.apply_along_axis(func, 1, U)  # 1D vector
        FEs += popsize
        
        mask = ( UFit <= popFit )   # bool 1D vector 
        popX = np.where(mask.reshape((popsize,1)), U, popX)
        popFit = np.where(mask, UFit, popFit)
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit  :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence)
            


def jDE(func, popsize:int = 100, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """Self-adapting control parameters in differential evolution

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
    Brest, Janez, et al. "Self-adapting control parameters in differential evolution: 
    A comparative study on numerical benchmark problems." IEEE transactions on evolutionary 
    computation 10.6 (2006): 646-657.
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
    
    popF  = rng.uniform(0.1, 0.9, (popsize) )  # 1D vector, --> Fl=0.1  Fu=0.9
    popCr = rng.uniform( 0,    1, (popsize) )  # 1D vector 
    
    while FEs <= MaxFEs: 
        maskTau1 = rng.uniform(0, 1, (popsize)) < 0.1  # --> tau1 = 0.1
        newpopF = np.where(maskTau1, rng.uniform(0.1, 0.9, (popsize)), popF)
        
        maskTau2 = rng.uniform(0, 1, (popsize)) < 0.1  # --> tau2 = 0.1
        newpopCr = np.where(maskTau2, rng.uniform( 0, 1, (popsize)), popCr)
        
        # -- Mutation --
        # DE/rand/1
        # three random indexes, they are not equal.
        a = np.ones( (popsize, popsize), dtype=int) * np.arange(0, popsize)
        idM = rng.permuted(a, axis=1)  # each slice along the given axis is shuffled independently of the others.
        idM = idM[:, 0:3]      # 1st 2nd 3rd columns are needed. 
        
        V = popX[ idM[:,0], :] + newpopF.reshape((popsize, 1)) * ( popX[ idM[:,1], :] - popX[ idM[:,2], :] ) # V.shape == popX.shape
        
        # -- Crossover --
        a = rng.uniform(0,1, (popsize, dim) )
        # ensure at least one dimension is exchanged
        b = np.zeros( (popsize, dim), dtype=bool)
        b[:,0] = True
        c = rng.permuted(b, axis=1) 
        # bool matrix for Crossover
        crossM = ( a < newpopCr.reshape((popsize, 1)) ) + c
        
        U = np.where(crossM, V, popX)  # U.shape == popX.shape
        U = np.clip(U, lb, ub)  # boundary control
        
        # -- Selection --
        UFit = np.apply_along_axis(func, 1, U)  # 1D vector
        FEs += popsize
        
        mask = ( UFit <= popFit )   # bool 1D vector 
        popX = np.where(mask.reshape((popsize,1)), U, popX)
        popFit = np.where(mask, UFit, popFit)
        popF  = np.where(mask, newpopF,  popF)
        popCr = np.where(mask, newpopCr, popCr)
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit  :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence)







