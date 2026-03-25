
import sys
sys.path.append('..')

import numpy as np
import npea

# should install opfunu package
from opfunu.cec_based import cec2017

cecfunList = [
cec2017.F12017(ndim=30),
cec2017.F22017(ndim=30),
cec2017.F32017(ndim=30),
cec2017.F42017(ndim=30),
cec2017.F52017(ndim=30),
cec2017.F62017(ndim=30),
cec2017.F72017(ndim=30),
cec2017.F82017(ndim=30),
cec2017.F92017(ndim=30),
cec2017.F102017(ndim=30),
cec2017.F112017(ndim=30),
cec2017.F122017(ndim=30),
cec2017.F132017(ndim=30),
cec2017.F142017(ndim=30),
cec2017.F152017(ndim=30),
cec2017.F162017(ndim=30),
cec2017.F172017(ndim=30),
cec2017.F182017(ndim=30),
cec2017.F192017(ndim=30),
cec2017.F202017(ndim=30),
cec2017.F212017(ndim=30),
cec2017.F222017(ndim=30),
cec2017.F232017(ndim=30),
cec2017.F242017(ndim=30),
cec2017.F252017(ndim=30),
cec2017.F262017(ndim=30),
cec2017.F272017(ndim=30),
cec2017.F282017(ndim=30),
cec2017.F292017(ndim=30)
]


gbestSol, gbestFit, convergence = npea.DE_.DE( cecfunList[1].evaluate, 
               popsize=100, F=0.5, Cr=0.4, 
               lb = [-100,]*30 , ub = [ 100,]*30 , MaxFEs=10000*30, seed=22) 

print('gbestSol: ' + str(gbestSol) )
print('gbestFit: ' + str(gbestFit) )

input("OK")


