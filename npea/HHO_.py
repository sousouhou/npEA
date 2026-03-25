"""
The implementation of different Harris hawks optimization algorithms.  

List of algorithms
------------------
    HHO      Original Harris hawks optimization     

"""
import numpy as np
import math
  
def HHO(func, popsize:int = 100, alphaLevy = 0.1,
    lb:list = None, ub:list = None,  
    MaxFEs:int = 100000, seed:int = 42):
    """The original Harris hawks optimization (HHO).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is 
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    alphaLevy : float, (0, 1)
        Scale factor of Levy flight.
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
    Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, 
    H. (2019). Harris hawks optimization: Algorithm and applications. Future 
    generation computer systems, 97, 849-872. 
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
    gbestSol = popX[bestId].copy()  # gbest solution, 1D vector, the location of Rabbit in HHO
    gbestFit = popFit[bestId]       # gbest fitness
    convergence = np.array( [[FEs, gbestFit]] ) 
    
    newpopX = popX.copy()
    newpopFit = popFit.copy()

    while FEs <= MaxFEs: 
    
        E1 = 2*( 1- (FEs/MaxFEs) ) # factor to show the decreaing energy of rabbit
        E = rng.uniform(-1, 1, size=(popsize) ) * E1   # 1D, for each hawks
    
        # -------- Exploration --------
        q = rng.uniform(0, 1, size=(popsize))
        randIds = rng.integers(0, popsize, size=(popsize))
        
        r1 = rng.uniform(0, 1, size=(popsize,1))
        r2 = rng.uniform(0, 1, size=(popsize,1))
        r3 = rng.uniform(0, 1, size=(popsize,1))
        r4 = rng.uniform(0, 1, size=(popsize,1))
        
        # -- perch based on other family members
        mask1 = (np.abs(E)>=1) * (q<0.5)   # '*' is AND
        if mask1.sum() > 0 : 
            # before mask : newpopX = popX[ randIds ] - r1*np.abs( popX[ randIds ] - 2*r2* popX)
            newpopX[mask1] = popX[ randIds[mask1] ] - r1[mask1] * np.abs( popX[ randIds[mask1] ] - 2*r2[mask1]* popX[mask1] )

        # -- perch on a random tall tree 
        mask2 = (np.abs(E)>=1) * (q>=0.5)   # '*' is AND
        if mask2.sum() > 0 : 
            # before mask : newpopX = (gbestSol -  popX.mean(axis=0)) - r3 *( lb + r4 * (ub-lb))
            newpopX[mask2] = (gbestSol - popX.mean(axis=0)) - r3[mask2] *( lb + r4[mask2] * (ub-lb))
    
        # -------- Exploitation --------
        r = rng.uniform(0, 1, size=(popsize))
        J = 2*(1 - rng.uniform(0, 1, size=(popsize,1)) )
        
        # ---- phase 1: surprise pounce (seven kills)
        # - Hard besiege
        mask3 = (np.abs(E)<1) * (r>=0.5) * (np.abs(E)<0.5)  
        if mask3.sum() > 0 : 
            # before mask : newpopX = gbestSol - E.reshape((popsize,1)) * np.abs(gbestSol - popX)
            newpopX[mask3] = gbestSol - E.reshape((popsize,1))[mask3] * np.abs(gbestSol - popX[mask3])
            
        # - Soft besiege
        mask4 = (np.abs(E)<1) * (r>=0.5) * (np.abs(E)>=0.5)  
        if mask4.sum() > 0 : 
            # before mask : newpopX = (gbestSol - popX) - E.reshape((popsize,1))  * np.abs( J * gbestSol - popX )
            newpopX[mask4] = (gbestSol - popX[mask4]) - E.reshape((popsize,1))[mask4]  * np.abs( J[mask4] * gbestSol - popX[mask4] )
        
        # - evaluate the above
        mask1234 = mask1 + mask2 + mask3 + mask4   # '+' is OR
        newpopX[mask1234] = np.clip(newpopX[mask1234], lb, ub)  # boundary control
        newpopFit[mask1234] = np.apply_along_axis(func, 1, newpopX[mask1234]) # 1D vector, len(popFit)==popsize
        FEs += mask1234.sum()

        # ---- phase 2: performing team rapid dives (leapfrog movements) 
        beta = 1.5 
        sigma = math.pow(math.gamma(1. + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2.) * beta * math.pow(2., (beta - 1) / 2)), 1. / beta)     
        u = rng.normal(0, 1, size=(popsize, dim))
        v = rng.normal(0, 1, size=(popsize, dim))
        LevyD = u*sigma / np.power(np.abs(v), 1.0/beta)  * alphaLevy * (ub-lb)
        
        # - Soft besiege + rapid dives
        mask5 = (np.abs(E)<1) * (r<0.5) * (np.abs(E)>=0.5)  
        if mask5.sum() > 0 : 
            # before mask : Y = gbestSol - E.reshape((popsize,1))  * np.abs( J * gbestSol - popX )
            Y = gbestSol - E.reshape((popsize,1))[mask5]  * np.abs( J[mask5] * gbestSol - popX[mask5] )   # shape is (mask5.sum(), dim)
            Z = Y + rng.uniform(0, 1, size=Y.shape) * LevyD[mask5]
            
            Y = np.clip(Y, lb, ub)  # boundary control
            Z = np.clip(Z, lb, ub)  # boundary control
            YFit = np.apply_along_axis(func, 1, Y)
            ZFit = np.apply_along_axis(func, 1, Z)
            FEs += (2*mask5.sum())
            
            bettermask = YFit<ZFit
            newpopX[mask5] = np.where( bettermask.reshape( (mask5.sum(),1) ), Y, Z)
            newpopFit[mask5] = np.where(bettermask, YFit, ZFit)
        
        # - Hard besiege + rapid dives
        mask6 = (np.abs(E)<1) * (r<0.5) * (np.abs(E)<0.5)   
        if mask6.sum() > 0 : 
            Y = gbestSol - E.reshape((popsize,1))[mask6]  * np.abs( J[mask6] * gbestSol - popX.mean(axis=0) )   # here is different from mask5 lines
            Z = Y + rng.uniform(0, 1, size=Y.shape) * LevyD[mask6]
            
            Y = np.clip(Y, lb, ub)  # boundary control
            Z = np.clip(Z, lb, ub)  # boundary control
            YFit = np.apply_along_axis(func, 1, Y)
            ZFit = np.apply_along_axis(func, 1, Z)
            FEs += (2*mask6.sum())
            
            bettermask = YFit<ZFit
            newpopX[mask6] = np.where( bettermask.reshape( (mask6.sum(), 1) ), Y, Z)    
            newpopFit[mask6] = np.where(bettermask, YFit, ZFit)
        
        popX   = newpopX.copy()
        popFit = newpopFit.copy()
            
        # --- Update gbest ---   
        bestId = np.argmin(popFit)
        if popFit[bestId] < gbestFit :
            gbestSol = popX[bestId].copy()
            gbestFit = popFit[bestId]
        
        if FEs%(int(MaxFEs/50)) < popsize*1.5 :
            print("FEs: %10d,  gbestFit: %.18e"%(FEs, gbestFit) )
            convergence = np.vstack( (convergence, np.array([[FEs, gbestFit]])) )
            
    return (gbestSol, gbestFit, convergence) 
            

