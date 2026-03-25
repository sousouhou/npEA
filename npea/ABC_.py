"""
The implementation of different artificial bee colony algorithms. 

List of algorithms
------------------
    ABC      Original artificial bee colony algorithm    
    GABC     Gbest-guided artificial bee colony algorithm
    MABC     Modified artificial bee colony algorithm
"""
import numpy as np

  
def ABC(func, popsize:int = 100, limit:int = 200,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original artificial bee colony (ABC).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    limit : int
        A food source will be abandoned if it cannot be improved.
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
    Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for 
    numerical function optimization: artificial bee colony (ABC) algorithm. Journal
    of global optimization, 39(3), 459-471.
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    # ---- Initialization ----
    # pop are the food-sources in ABC, popsize is SN in ABC
    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize
    
    trial = np.zeros(popsize) # 1D

    bestId  = np.argmin(popFit)     
    gbestSol = popX[bestId].copy()   # best solution, 1D vector
    gbestFit = popFit[bestId]        # best fitness
    convergence = np.array( [[FEs, gbestFit]] ) 
    
    while FEs <= MaxFEs: 
    
        # ---- employed bee phase ----
        # each X selection a pair, their indexs are not equal
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame = ( pairId==np.arange(0, popsize) )
        pairId[masksame] = (pairId[masksame] +1) % popsize  
    
        phi = rng.uniform(-1,1, (popsize, dim) )
        tempV = popX + phi * (popX - popX[pairId]) 
        # ensure only one dimension is exchanged
        b = np.zeros( (popsize, dim), dtype=bool)
        b[:,0] = True
        c = rng.permuted(b, axis=1)   
        V = np.where(c, tempV, popX)
        V = np.clip(V, lb, ub)  # boundary control
        
        VFit = np.apply_along_axis(func, 1, V)  # 1D vector
        FEs += popsize
        
        improved = VFit < popFit   
        popX[improved] = V[improved]
        popFit[improved] = VFit[improved]
        
        trial[~improved] += 1
        trial[improved]   = 0
        
        # ----- onlooker bee phase -----
        # popFit -> fitness, popFit may be negative, fitness must be positive
        temp1 = 1.0/( 1+popFit )
        temp2 = 1.0 + np.absolute(popFit) 
        fitness = np.where( popFit>=0, temp1, temp2 )
    
        # roulette wheel selection for onlookers
        # choose a food source depending on its probability to be chosen 
        probs = ( fitness) / fitness.sum()
                    # a food source can be selected several times. replace=True, can be selected multiple times.
        selectedId = rng.choice(np.arange(0, popsize), size=popsize, replace=True, p=probs)     
  
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame =  ( pairId== selectedId )
        pairId[masksame] = (pairId[masksame] +1) % popsize     # confirm their pair
        
        for i in range(0, popsize):  # can not use matrix for parallel computing, must use one for-loop
            id1 = selectedId[i]
            id2 = pairId[i]
            
            phi = rng.uniform(-1, 1)
            tt = rng.integers(0, dim) # target dimension
            V = popX[id1].copy()
            V[tt] = popX[id1][tt] + phi * (popX[id1][tt] - popX[id2][tt])  #1D
            V = np.clip(V, lb, ub)    # boundary control
            
            VFit = func(V)
            FEs += 1
            
            if( VFit < popFit[id1]) :
                popX[id1] = V[:]
                popFit[id1] = VFit
                trial[id1] = 0
            else :
                 trial[id1] += 1
                 
        # --- scout bee phase ---
        scoutId = np.where(trial >= limit)[0]   # 1D vector
        if len(scoutId) > 0:
            popX[scoutId] = rng.uniform(lb, ub, (len(scoutId), dim))  # 2D matrix
            popFit[scoutId] = np.apply_along_axis(func, 1, popX[scoutId]) 
            FEs += len(scoutId)    
            trial[scoutId] = 0
            
        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*2 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )   
            
    return (gbestSol, gbestFit, convergence) 
    
    
    
    
def GABC(func, popsize:int = 100, limit:int = 200, C:float = 1.5, 
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The Gbest-guided artificial bee colony algorithm.

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    limit : int
        A food source will be abandoned if it cannot be improved.
    C : float
        A nonnegative constant.
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
    Zhu, Guopu, and Sam Kwong. "Gbest-guided artificial bee colony algorithm for
    numerical function optimization." Applied mathematics and computation 217.7 
    (2010): 3166-3173.
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    # ---- Initialization ----
    # pop are the food-sources in ABC, popsize is SN in ABC
    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize
    
    trial = np.zeros(popsize) # 1D

    bestId  = np.argmin(popFit)     
    gbestSol = popX[bestId].copy()   # best solution, 1D vector
    gbestFit = popFit[bestId]        # best fitness
    convergence = np.array( [[FEs, gbestFit]] ) 
    
    while FEs <= MaxFEs: 
    
        # ---- employed bee phase ----
        # each X selection a pair, their indexs are not equal
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame = ( pairId==np.arange(0, popsize) )
        pairId[masksame] = (pairId[masksame] +1) % popsize  
    
        phi = rng.uniform(-1, 1, (popsize, dim) )
        psi = rng.uniform( 0, C, (popsize, dim) )
        tempV = popX + phi * (popX - popX[pairId]) + psi * ( gbestSol - popX)
        # ensure only one dimension is exchanged
        b = np.zeros( (popsize, dim), dtype=bool)
        b[:,0] = True
        c1 = rng.permuted(b, axis=1)   
        V = np.where(c1, tempV, popX)
        V = np.clip(V, lb, ub)  # boundary control
        
        VFit = np.apply_along_axis(func, 1, V)  # 1D vector
        FEs += popsize
        
        improved = VFit < popFit   
        popX[improved] = V[improved]
        popFit[improved] = VFit[improved]
        
        trial[~improved] += 1
        trial[improved]   = 0
        
        # ----- onlooker bee phase -----
        # popFit -> fitness, popFit may be negative, fitness must be positive
        temp1 = 1.0/( 1+popFit )
        temp2 = 1.0 + np.absolute(popFit) 
        fitness = np.where( popFit>=0, temp1, temp2 )
    
        # roulette wheel selection for onlookers
        # choose a food source depending on its probability to be chosen 
        probs = ( fitness) / fitness.sum()
                    # a food source can be selected several times. replace=True, can be selected multiple times.
        selectedId = rng.choice(np.arange(0, popsize), size=popsize, replace=True, p=probs)     
  
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame =  ( pairId== selectedId )
        pairId[masksame] = (pairId[masksame] +1) % popsize     # confirm their pair
        
        for i in range(0, popsize):  # can not use matrix for parallel computing, must use one for-loop
            id1 = selectedId[i]
            id2 = pairId[i]
            
            phi = rng.uniform(-1, 1)
            psi = rng.uniform( 0, C)
            tt = rng.integers(0, dim) # target dimension
            V = popX[id1].copy()
            V[tt] = popX[id1][tt] + phi * (popX[id1][tt] - popX[id2][tt]) + psi * ( gbestSol[tt] - popX[id1][tt])
            V = np.clip(V, lb, ub)    # boundary control
            
            VFit = func(V)
            FEs += 1
            
            if( VFit < popFit[id1]) :
                popX[id1] = V[:]
                popFit[id1] = VFit
                trial[id1] = 0
            else :
                 trial[id1] += 1
                 
        # --- scout bee phase ---
        scoutId = np.where(trial >= limit)[0]   # 1D vector
        if len(scoutId) > 0:
            popX[scoutId] = rng.uniform(lb, ub, (len(scoutId), dim))  # 2D matrix
            popFit[scoutId] = np.apply_along_axis(func, 1, popX[scoutId]) 
            FEs += len(scoutId)    
            trial[scoutId] = 0
            
        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*2 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )   
            
    return (gbestSol, gbestFit, convergence) 
    


def MABC(func, popsize:int = 100, limit:int = 200, MR:float = 0.4, SF:float =1.0,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The modified artificial bee colony (MABC).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    limit : int
        A food source will be abandoned if it cannot be improved.
    MR : float
        Modification rate
    SF : float
        Scaling factor
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
    Akay, Bahriye, and Dervis Karaboga. "A modified artificial bee colony algorithm 
    for real-parameter optimization." Information sciences 192 (2012): 120-142.
    """
    print(locals()) # print parameters
    FEs = 0
    
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    # ---- Initialization ----
    # pop are the food-sources in ABC, popsize is SN in ABC
    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize
    
    trial = np.zeros(popsize) # 1D

    bestId  = np.argmin(popFit)     
    gbestSol = popX[bestId].copy()   # best solution, 1D vector
    gbestFit = popFit[bestId]        # best fitness
    convergence = np.array( [[FEs, gbestFit]] ) 
    
    while FEs <= MaxFEs: 
    
        # ---- employed bee phase ----
        # each X selection a pair, their indexs are not equal
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame = ( pairId==np.arange(0, popsize) )
        pairId[masksame] = (pairId[masksame] +1) % popsize  
    
        phi = rng.uniform(-1,1, (popsize, dim) ) * SF   #modified
        tempV = popX + phi * (popX - popX[pairId]) 
        # ensure MR dimensions are exchanged
        c = rng.uniform(0, 1, (popsize, dim)) < MR
        V = np.where(c, tempV, popX)
        V = np.clip(V, lb, ub)  # boundary control
        
        VFit = np.apply_along_axis(func, 1, V)  # 1D vector
        FEs += popsize
        
        improved = VFit < popFit   
        popX[improved] = V[improved]
        popFit[improved] = VFit[improved]
        
        trial[~improved] += 1
        trial[improved]   = 0
        
        # ----- onlooker bee phase -----
        # popFit -> fitness, popFit may be negative, fitness must be positive
        temp1 = 1.0/( 1+popFit )
        temp2 = 1.0 + np.absolute(popFit) 
        fitness = np.where( popFit>=0, temp1, temp2 )
    
        # roulette wheel selection for onlookers
        # choose a food source depending on its probability to be chosen 
        probs = ( fitness) / fitness.sum()
                    # a food source can be selected several times. replace=True, can be selected multiple times.
        selectedId = rng.choice(np.arange(0, popsize), size=popsize, replace=True, p=probs)     
  
        pairId = rng.permuted( np.arange(0, popsize) )  # 1D vector
        masksame =  ( pairId== selectedId )
        pairId[masksame] = (pairId[masksame] +1) % popsize     # confirm their pair
        
        for i in range(0, popsize):  # can not use matrix for parallel computing, must use one for-loop
            id1 = selectedId[i]
            id2 = pairId[i]
            
            phi = rng.uniform(-1, 1) * SF   # modified
            mask01 =  rng.uniform(0, 1, (dim)) < MR   # modified
            V = popX[id1].copy()
            V[mask01] = popX[id1][mask01] + phi * (popX[id1][mask01] - popX[id2][mask01])  # modified
            V = np.clip(V, lb, ub)    # boundary control
            
            VFit = func(V)
            FEs += 1
            
            if( VFit < popFit[id1]) :
                popX[id1] = V[:]
                popFit[id1] = VFit
                trial[id1] = 0
            else :
                 trial[id1] += 1
                 
        # --- scout bee phase ---
        scoutId = np.where(trial >= limit)[0]   # 1D vector
        if len(scoutId) > 0:
            popX[scoutId] = rng.uniform(lb, ub, (len(scoutId), dim))  # 2D matrix
            popFit[scoutId] = np.apply_along_axis(func, 1, popX[scoutId]) 
            FEs += len(scoutId)    
            trial[scoutId] = 0
            
        # --- update gbest ---
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*2 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )   
            
    return (gbestSol, gbestFit, convergence) 
    
    