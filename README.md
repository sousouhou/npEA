# npEA
- **Current version = 1.0.2**

### Overview

- npEA is an open-source Python package that implements a suite of Evolutionary Algorithms. 
- each EA in npEA fully leverages NumPy's vectorized operations and high-level array functions, resulting in improved efficiency and readability. 
- npEA employs a low-coupling design and maintains concise code, making it easy to extend. 


### Dependencies and Installation
- Python (>= 3.10.1)
- NumPy (>=1.26.0)

Use pip to install Dependencies, such as:
```
pip install numpy==1.26.0
```

Use pip to install npEA with a specific version:
```
pip install npea==1.0.2
```


### Usage

####  1. See documentation string for help
- View npEA's docstring:
```Python
>>> import npea
>>> help(npea)  

```

- View a module's docstring (e.g., the PSO_ module):
```Python
>>> import npea
>>> help(npea.PSO_)   

```

- View a function's docstring (e.g., the PSO function):
```Python
>>> import npea
>>> help(npea.PSO_.PSO)    
Help on function PSO in module npea.PSO_:

PSO(func, popsize: int = 100, w=0.5, c1=1.5, c2=1.5, lb: list = None, ub: list = None, MaxFEs: int = 100000, seed: int = 42)
    The standard particle swarm optimization (PSO).

    Parameters
    ----------
    func : callable
        Objective function for minimum optimization. Signature is
        ``func(X: np.ndarray) -> float``.
    popsize : int
        Population size.
    w : float
        Inertia weight.
    c1 : float
        Cognitive weight.
    c2 : float
        Social weight.
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
    Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization.
    In Proceedings of ICNN'95-international conference on neural networks (Vol. 4,
    pp. 1942-1948). ieee.
    Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer."
    Evolutionary computation proceedings. Vol. 890. 1998.
```

#### 2. A simple example of optimization

The code to solve a problem using the original DE algorithm is shown as follows.
```Python
import numpy as np
import npea

def myObjectivefunction(solution):   # minimization
    ndim = solution.shape[0]         # if necessary
    return np.sum( (solution-9.16)**2)

gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction, 
               popsize=100, F=0.5, Cr=0.9, 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2) 
    
print('gbestSol: ' + str(gbestSol) )
print('gbestFit: ' + str(gbestFit) )
print('convergence: \n' + str(convergence) )      
```
```
Returns
-------
gbestSol : 1D ndarray
    Global best solution found by this algorithm.
gbestFit: float
    Objective function value of gbestSol.
convergence : 2D ndarray
    Convergence history, shape is (~50, 2), the 1st column is FEs, the 2nd column is gbestFit.
```


#### 3. Evaluation with data
The objective function may require additional data to evaluate a solution. 
An example of training a linear regression model using npEA is shown below.    
```Python
import numpy as np
import npea
# The data should be loaded outside the scope of myObjectivefunction.
trainX = np.array([[1.0, 2.0],
                   [3.0, 4.0],
                   [6.0, 6.0],
                   [7.0, 7.0]])
trainY = np.array([10.0, 11.0, 12.0, 13.0])     

def myObjectivefunction(solution):   # minimization
    w = solution[0:2]  # decode
    b = solution[2]
    # Do not load data here; otherwise, it will slow down execution.
    predictY = trainX.dot(w) + b
    return np.sum( ( trainY - predictY )**2 )  # mean square error
    
gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction, 
               popsize=100, F=0.5, Cr=0.9, 
               lb = [-100,]*3 , ub = [ 100,]*3 , MaxFEs=10000, seed=12) 
print('gbestSol: ' + str(gbestSol) )   
```


### List of algorithms in npEA 
|  Module  name     | Function name |  Description                                          |
|    :----:         |   :----:      |       :----                                           |
|   ABC_.py         |    ABC        |  Artificial bee colony                                |
|                   |    GABC       |  Gbest-guided artificial bee colony                   |
|                   |    MABC       |  Modified artificial bee colony                       |
|   BA_.py          |    BA         |  Bees Algorithm                                       |
|   BBO_.py         |    BBO        |  Biogeography-Based Optimization                      |  
|   CS_.py          |    CS         |  Cuckoo search                                        |  
|   CSO_.py         |    CSO        |  Competitive swarm optimizer                          |  
|   DE_.py          |    DE         |  Differential Evolution                               |  
|                   |    DEmu       |  Differential Evolution with different mutation strategies |  
|                   |    jDE        |  Self-adapting control parameters in differential evolution      |  
|   EDA_.py         |    EDA        |  Estimation of distribution algorithm                 |  
|   ES_.py          |    ES         |  Evolution Strategy                                   |  
|   FA_.py          |    FA         |  Firefly Algorithm                                    |  
|   FWA_.py         |    FWA        |  Fireworks Algorithm                                  |  
|   GA_.py          |    GA         |  Genetic Algorithm                                    |  
|   GSA_.py         |    GSA        |  Gravitational Search Algorithm                       |  
|   GWO_.py         |    GWO        |  Grey Wolf Optimizer                                  |  
|   HHO_.py         |    HHO        |  Harris hawks optimization                            |  
|   PSO_.py         |    PSO        |  Particle swarm optimization                          |  
|                   |    DIPSO      |  Decreasing Inertia Particle Swarm Optimization       |  
|   SA_.py          |    SA         |  Simulated Annealing                                  |  
|   SCA_.py         |    SCA        |  Sine cosine algorithm                                |  
|                   |    SCAgreedy  |  SCA with greedy strategy                             |  
|   TLBO_.py        |    TLBO       |  Teaching–learning-based optimization                 |  
|   TS_.py          |    TS         |  Tabu Search                                          |  
|   WOA_.py         |    WOA        |  Whale Optimization Algorithm                         |  



### Changelog 

- Version 1.0.2
Modified version number.

- Version 1.0.1
Releasing the package. 


### Citations
Under review.


### LICENSE
npEA is available under the Apache License.


