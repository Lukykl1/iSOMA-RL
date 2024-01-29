import opfunu
import numpy as np
import os
import time
import random
import pyade.ilshade
import pyade.mpede


def run_solver(model, name, problem_definition, dim, year, func):
    _, best_fitness = model.apply(**problem_definition)
    print(f"Fitness: {best_fitness}, Difference: {best_fitness - func.f_global}")
    with open("log3/" + year + "/" + str(dim) + "/" + name + ".txt", "a") as f:
        f.write(year + "" + func.name + "\t" + str(best_fitness - func.f_global) + "\n")


i = 0


def Wrapper(x, func):
    global i
    i += 1
    # if i % 1000 == 0:
    # print(i)
    return func(x)


benchmark = ["2015", "2017", "2013"]
random.shuffle(benchmark)
os.makedirs("log3", exist_ok=True)
needed = [
    "F22: Composition Function 3",
    "F20: Composition Function 1",
    "F18: Hybrid Function 9",
    "F25: Composition Function 6",
    "F2: Shifted and Rotated Zakharov Function",
    "F28: Composition Function 9",
    "F21: Composition Function 2",
    "F23: Composition Function 4",
    "F26: Composition Function 7",
    "F19: Hybrid Function 10",
    "F27: Composition Function 8",
    "F24: Composition Function 5",
]
for year in benchmark:
    os.makedirs("log3/" + year, exist_ok=True)
    print(year)
    dimensions = [10, 30]
    for dim in dimensions:
        os.makedirs("log3/" + year + "/" + str(dim), exist_ok=True)
        print(dim)
        functions = opfunu.get_functions_based_classname(year)
        random.shuffle(functions)
        for funcType in functions:
            func = funcType(ndim=dim)
            # if func.name not in needed:
            #    continue
            print(year, dim, func.name)
            # ilshade
            params = pyade.ilshade.get_default_params(dim=dim)
            params["max_evals"] = dim * 10**4
            params["bounds"] = np.array(func.bounds)
            params["func"] = lambda x: Wrapper(x, func.evaluate)
            print(params)
            run_solver(pyade.ilshade, "ilshade", params, dim, year, func)
            # mpede
            params = pyade.mpede.get_default_params(dim=dim)
            params["max_evals"] = dim * 10**4
            params["bounds"] = np.array(func.bounds)
            params["func"] = lambda x: Wrapper(x, func.evaluate)
            run_solver(pyade.mpede, "MPEDE", params, dim, year, func)
