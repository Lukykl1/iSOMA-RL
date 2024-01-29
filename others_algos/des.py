from scipy.optimize import differential_evolution
import opfunu
import numpy as np

for funcType in opfunu.get_functions_based_classname('2017'):
    for i in range(20):
        func = funcType(ndim=30)
        bounds = func.bounds
        result = differential_evolution(func.evaluate, bounds, maxiter=30*10**4)
        print(str(i) + str(funcType.name)+ "\t" + str(result.x) + "\t" + str(result.fun) + "\t"+ str(func.f_global - result.fun) + "\n")
        f = open("results_des_2017.txt", "a")
        f.write(str(funcType.name)+ "\t" + str(abs(func.f_global - result.fun))  + "\n")
