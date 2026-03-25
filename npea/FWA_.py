"""
The implementation of different Fireworks Algorithms.

List of algorithms
------------------
    FWA      Original Fireworks Algorithm     

"""
import numpy as np

  
def FWA(func, popsize:int = 5, m:int = 50, mhat:int = 5, a = 0.04, b = 0.8, Ahat = 0.4,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Fireworks Algorithm (FWA).
        
    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    m : int
        Number of common sparks.
    mhat : int
        Number of Gaussian sparks.
    a : float
        A const parameter.
    b : float
        A const parameter.
    Ahat : float, [0,1] 
        Max explosion amplitude (relative)
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
    Tan, Y., & Zhu, Y. (2010, June). Fireworks algorithm for optimization. 
    In International conference in swarm intelligence (pp. 355-364). Berlin, 
    Heidelberg: Springer Berlin Heidelberg.
    """
    assert ( (0 < a) and (a<b) and (b<1) )
    assert ( (0<Ahat) and (Ahat<1) )

    print(locals()) # print parameters
    FEs = 0
   
    lb = np.array(lb)
    ub = np.array(ub)
    rng = np.random.default_rng(seed)
    dim = lb.shape[0]

    popX = rng.uniform(lb, ub, (popsize, dim))  # 2D matrix, Fireworks
    popFit = np.apply_along_axis(func, 1, popX) # 1D vector, len(popFit)==popsize
    FEs += popsize

    bestId  = np.argmin(popFit)    
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector
    gbestFit = popFit[bestId]       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] )    
    
    while FEs <= MaxFEs: 
    
        sparksX   = np.zeros((1,dim))  # the first element is not necessary
        sparksFit = np.zeros((1,))
    
        fitmin = np.min(popFit)
        fitmax = np.max(popFit)
        
        # --- Common sparks ---
        for i in range(0, popsize):
            
            si = m * ( fitmax-popFit[i] + 1e-15) / ( np.sum(fitmax-popFit) + 1e-15)
            si = np.clip( si , a*m, b*m  )
            si = int( np.round(si) ) # scalar value
        
            Ai = Ahat * ( popFit[i]-fitmin + 1e-15) / ( np.sum(popFit-fitmin) + 1e-15) # scalar value
            
            # generate si sparks
            temp1 = popX[i] + Ai * (ub-lb) * rng.uniform(-1,1, (si, dim) )  # shape is (si, dim) 
            maskdim  =  rng.uniform(0,1, (si, dim) ) < 0.5  # Randomly select z dimensions, 
            gensparksX = np.where(maskdim, temp1, popX[i]*np.ones((si, dim)) )  # generated sparks
            
            # gensparksX = np.clip(gensparksX, lb, ub)     # boundary control
            maskexceed = (gensparksX < lb) + (gensparksX > ub)  #OR
            temp2 = lb + np.abs(gensparksX)%(ub-lb)   # boundary control
            gensparksX = np.where(maskexceed, temp2, gensparksX)
            
            gensparksFit = np.apply_along_axis(func, 1, gensparksX) 
            FEs += si            
        
            sparksX = np.vstack( (sparksX, gensparksX) )
            sparksFit = np.hstack( (sparksFit, gensparksFit) )
    
        # --- Gaussian sparks ---
        selIds = rng.choice(np.arange(0, popsize), size =mhat, replace=True)  # replace=True, can be selected multiple times.
        
        g = rng.normal(1, 1, size=(mhat,dim)) # Gaussian distribution, mu = 1, sigma = 1.     
        gensparksX = popX[selIds] * g         # OK, it seems to have a logical problem for negative values
    
        # gensparksX = np.clip(gensparksX, lb, ub)  # boundary control
        maskexceed = (gensparksX < lb) + (gensparksX > ub)  #OR
        temp2 = lb + np.abs(gensparksX)%(ub-lb)   # boundary control
        gensparksX = np.where(maskexceed, temp2, gensparksX)
        
        gensparksFit = np.apply_along_axis(func, 1, gensparksX) 
        FEs += mhat            
    
        sparksX = np.vstack( (sparksX, gensparksX) )
        sparksFit = np.hstack( (sparksFit, gensparksFit) )        

        sparksX = sparksX[1:]      # remove the first unnecessary element
        sparksFit = sparksFit[1:]
    
        # --- Selection ---
        # merge
        popX = np.vstack( (popX, sparksX) )
        popFit = np.hstack( (popFit, sparksFit) ) 
        
        numpopX = popX.shape[0]    # current size
                                                # diff is (numpopX, numpopX , dim),
        diff = np.reshape(popX, (1, numpopX, dim) ) - np.reshape(popX, (numpopX, 1 , dim) ) 
        Rmatrix = np.linalg.norm(diff, ord=2, axis=2)  # Rmatrix is （numtempX, numtempX)
        Rall = Rmatrix.sum(axis=1)
        probs = Rall / Rall.sum()  # Wheel selection     
        
        selectedIds = rng.choice(np.arange(0, numpopX), size=popsize, replace=False, p=probs) # replace=True, can be selected multiple times.
        
        bestId = np.argmin(popFit)
        if bestId not in selectedIds :
            selectedIds = np.hstack( ([bestId], selectedIds[0:-1] ) )
        
        popX = popX[selectedIds].copy()
        popFit = popFit[selectedIds].copy()
        
        # --- update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < m*2  :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) ) 
            
    return (gbestSol, gbestFit, convergence) 
            

