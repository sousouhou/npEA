"""
npEA
======
npEA: A NumPy-based Python package that implements vectorized evolutionary
algorithms.

npEA provides a suite of evolutionary algorithms implemented with 
vectorized NumPy code, ensuring speed, modularity, and clarity in a 
lightweight Python package.
    
Documentation is available in the docstrings provided with the code. 
The homepage of npEA is https://github.com/sousouhou/npEA.

Use the built-in "help" function to view a docstring:
>>> import npea
>>> help(npea)           # see npea's docstring
>>> help(npea.PSO_)      # see a module's docstring
>>> help(npea.PSO_.PSO)  # see a function's docstring

Examples
--------
>>> import numpy as np
>>> import npea
>>> 
>>> def myObjectivefunction(solution):  # minimization; solution is 1D ndarray
...     ndim = solution.shape[0]        # if necessary
...     return np.sum( (solution-9.16)**2)
>>> 
>>> gbestSol, gbestFit, convergence = npea.DE_.DE(myObjectivefunction,
...     popsize=100, F=0.5, Cr=0.9,
...     lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=2)
>>> 
>>> print('gbestSol: ' + str(gbestSol) )
>>> print('gbestFit: ' + str(gbestFit) )
>>> print('convergence: ' + str(convergence) )

"""

__version__ = "1.0.2"

# import all modules
import npea.ABC_ 
import npea.BA_ 
import npea.BBO_ 
import npea.CSO_ 
import npea.CS_ 
import npea.DE_ 
import npea.EDA_ 
import npea.ES_ 
import npea.FA_ 
import npea.FWA_ 
import npea.GA_ 
import npea.GSA_ 
import npea.GWO_ 
import npea.HHO_ 
import npea.PSO_ 
import npea.SA_ 
import npea.SCA_ 
import npea.TLBO_ 
import npea.TS_ 
import npea.WOA_ 










