import numpy as np
from functions import *
import opfunu
import random

def Ackley(x):
## Ackley's function. VarMin, VarMax = -1     , 1
  m, n = np.shape(x)
  f = np.zeros(n)
  a = 20.0
  b = 0.2
  c = 0.2
  for j in range(0, n):
    f[j] = - a * np.exp(-b*np.sqrt(np.sum(x[0:m,j]**2) / float(m))) \
      - np.exp(np.sum(np.cos(c*np.pi*x[0:m,j])) / float(m)) \
      + a + np.exp(1.0)
  return f
def Rosenbrock(x):
## Rosenbrock's valley. VarMin, VarMax = -2.048 , 2.048
  m, n = np.shape(x)
  f = np.zeros(n)
  for j in range(0,n):
    f[j] = np.sum((1.0-x[0:m,j])**2) \
         + np.sum((x[1:m,j]-x[0:m-1,j])**2)
  return f
def Schwefel(x):
## Schwefel's function. VarMin, VarMax = -500   , 500
  m, n = np.shape(x)
  f = np.zeros(n)
  for j in range(0, n):
    f[j] = 418.982887*m-np.sum(x[0:m,j]*np.sin(np.sqrt(abs(x[0:m,j]))))
  return f

allowed = ["F22: Composition Function 3",
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
benchmark = [ '2015', '2017', '2013']
random.shuffle(benchmark)
for year in benchmark:
    functions = opfunu.get_functions_based_classname(year)
    random.shuffle(functions)
    for funcType in functions:
      func = funcType(ndim=10)
      if func.name not in allowed:
        continue
      CostFunction = lambda x: Wrapper(x, func.evaluate) 
      print(func.name)
      def initialize_population(VarMin, VarMax, PopSize, dimension):
        # Create the initial population
        pop = VarMin + np.random.rand(dimension, PopSize) * (VarMax - VarMin)

        # Evaluate the initial population
        fitness = CostFunction(pop)
        the_best_cost = min(fitness)
        FEs = PopSize

        # Initialize other variables
        best_cost_old = the_best_cost

        return pop, fitness, FEs, the_best_cost, best_cost_old

      def migrate_to_leader(Migrant, Leader, N_jump, Step, Max_FEs, VarMin, VarMax, FEs, dimension, move):
          nstep = (N_jump-move+1) * Step
          # Update control parameters and perform mutation
          PRT = 0.1 + 0.9*(FEs / Max_FEs)
          PRTVector = (np.random.rand(dimension,1)<PRT)*1
          offspring = Migrant + (Leader - Migrant) * nstep * PRTVector
          # Check and put individuals inside the search range if it's outside
          for rw in range(dimension):
            if offspring[rw]<VarMin or offspring[rw]>VarMax:
              offspring[rw] = VarMin + np.random.rand() * (VarMax - VarMin)
          return offspring
        
      def evaluation_and_update(pop, fitness, M, M_sort, j, offspring, CostFunction, the_best_cost, the_best_value, FEs, Count, flag):
        # Evaluate the offspring and update the population
        new_cost = CostFunction(offspring)
        FEs = FEs + 1
        # Place the best offspring in the population
        if new_cost <= fitness[M[M_sort[j]]]:
          flag = 1
          fitness[M[M_sort[j]]] = new_cost
          pop[:, [M[M_sort[j]]]] = offspring
          if new_cost < the_best_cost:
            the_best_cost = new_cost
            the_best_value = offspring
        else:
          Count = Count + 1
        return pop, fitness, the_best_cost, the_best_value, FEs, flag

      def self_organizing(pop, fitness, dimension, m, n, k, N_jump, Step, Max_FEs, VarMin, VarMax, CostFunction, the_best_cost, the_best_value, FEs, Count, PopSize, best_cost_old):
        # Migrant selection: m
        M = np.random.choice(range(PopSize),m,replace=False)
        M_sort = np.argsort(fitness[M])
        for j in range(n):
          # Get the Migrant position (solution values) in the current population
          Migrant = pop[:, M[M_sort[j]]].reshape(dimension, 1)

          # Leader selection: k
          K = np.random.choice(range(PopSize),k,replace=False)
          K_sort = np.argsort(fitness[K])
          Leader = pop[:, K[K_sort[1]]].reshape(dimension, 1)
          if M[M_sort[j]] == K[K_sort[1]]:
            Leader = pop[:, K[K_sort[2]]].reshape(dimension, 1)

          # Perform self-organizing process on the Migrant  flag, move = 0, 1
          flag, move = 0, 1
          while (flag == 0) and (move <= N_jump) and FEs < Max_FEs:
              offspring = migrate_to_leader(Migrant, Leader, N_jump, Step, Max_FEs, VarMin, VarMax, FEs, dimension, move)
              pop, fitness, the_best_cost, the_best_value, FEs, flag = evaluation_and_update(pop, fitness, M, M_sort, j, offspring, CostFunction, the_best_cost, the_best_value, FEs, Count, flag) 
              # Replace migrants if the optimization process is stuck
              move = move + 1
        pop, fitness, Count, best_cost_old, FEs = migrants_replacement(pop, fitness, Count, PopSize, dimension, VarMin, VarMax, CostFunction, the_best_cost, best_cost_old, FEs)
          
        return pop, fitness, the_best_cost, the_best_value, FEs, Count, best_cost_old

      def migrants_replacement(pop, fitness, Count, PopSize, dimension, VarMin, VarMax, CostFunction, the_best_cost, best_cost_old, FEs):
        if Count > PopSize * 50:
          if the_best_cost == best_cost_old:
            rat = round(0.1 * PopSize)
            pop_temp = VarMin + np.random.rand(dimension, rat)*(VarMax-VarMin)
            fit_temp = CostFunction(pop_temp)
            FEs = FEs + rat
            D = np.random.choice(range(PopSize),rat,replace=False)
            pop[:,D] = pop_temp
            fitness[D] = fit_temp
          else:
            best_cost_old = the_best_cost
          Count = 0
        return pop, fitness, Count, best_cost_old, FEs

      def run_isoma(dimension, N_jump, Step, PopSize, Max_FEs, VarMin, VarMax, m, n, k): 
        #Initialize the population
        pop, fitness, FEs, the_best_cost, best_cost_old = initialize_population(VarMin, VarMax, PopSize, dimension)
        Count = 0
        Migration = 0
        the_best_value = None
        while FEs < Max_FEs:              
          Migration = Migration + 1                  
          # Perform self-organizing process on the population
          pop, fitness, the_best_cost, the_best_value, FEs, Count, best_cost_old = self_organizing(pop, fitness, dimension, m, n, k, N_jump, Step, Max_FEs, VarMin, VarMax, CostFunction, the_best_cost, the_best_value, FEs, Count, PopSize, best_cost_old)
        
        return the_best_cost, the_best_value, FEs, Migration



      the_best_cost, the_best_value, FEs, Migration = run_isoma(dimension=10, N_jump=10, Step=0.3, PopSize=100, Max_FEs=10*10**4 , VarMin=-func.bounds[0][1], VarMax=func.bounds[0][1], m=10, n=5, k=15)
      print('Stop at Migration :  ', Migration)
      print('The number of FEs :  ', FEs)
      print('The best cost     :  ', the_best_cost)
      print('Solution values   :  ', the_best_value)
      print('Differece         :  ', the_best_cost - func.f_global)

      #write to log/isoma_results.txt
      with open('log/log_orig32.txt', 'a') as f:
        f.write(year+func.name + "\t" +  str(the_best_cost[0] - func.f_global) + '\n')
