
import gym
import numpy as np
from gym.spaces import Dict, Discrete, Box
import os
import sys
import gym
import numpy as np
import re

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, 
     PopSize, dimension, CostFunctions,
      Max_FEs,
      m, n, k, HistoryLen, f_global, x_global, function_name, write, pretrained, log_dir_results
      ):
        super().__init__()

        # replacement, move, step
        self.action_space = Box(low=np.array([-1]), high=np.array([1]), shape=(1,))
        
        self.HistoryLen = HistoryLen
        self.observation_space = Dict({
            "history": Box(shape=(self.HistoryLen,1), low=-float('inf'), high=float('inf')),
            "difference_history": Box(shape=(self.HistoryLen,1), low=-float('inf'), high=float('inf')),
            "action": Box(shape=(self.HistoryLen,1), low=-float('inf'), high=float('inf')),
            })
        
        self.Max_FEs = Max_FEs
        self.dimension = dimension
        self.CostFunctions = CostFunctions
        selected = self.CostFunctions[np.random.randint(0, len(self.CostFunctions))]
        self.CostFunction = selected[0]
        self.VarMin = -selected[1]
        self.VarMax = selected[1]
        self.PopSize = PopSize
        self.m = m
        self.n = n
        self.k = k
        self.f_global = f_global
        self.x_global = x_global
        self.function_name = function_name
        self.write = write
        self.total_migrations = 0
        self.total_difference = 1
        self.pretrained = pretrained    
        self.log_dir_results = log_dir_results

        self.the_best_value = None
        self.Count = 0
        self.Migration = 0

        self.pop = self.VarMin + np.random.rand(self.dimension, self.PopSize) * (self.VarMax - self.VarMin)

        # Evaluate the initial population
        self.fitness = self.CostFunction(self.pop)
        self.the_best_cost = min(self.fitness)
        self.history = []
        self.FEs = PopSize

        # Initialize other variables
        self.history = [self.the_best_cost]
        self.difference_history = [self.the_best_cost]
        self.action = [0]
        self.best_cost_old = self.the_best_cost
        
        self.prtFunc = lambda x : (0.35 + 0.25 * x[0])
    
    def migrate_to_leader(self, Migrant, Leader, move, action):
        #
        # Update control parameters and perform mutation
        nstep = (round(10)-move+1) * (0.3)
        PRT = self.prtFunc(action) + 0.2*(self.FEs / self.Max_FEs)

        PRTVector = (np.random.rand(self.dimension,1)<PRT)*1
        offspring = Migrant + (Leader - Migrant) * nstep * PRTVector
        # Check and put individuals inside the search range if it's outside
        for rw in range(self.dimension):
          if offspring[rw]<self.VarMin or offspring[rw]>self.VarMax:
            offspring[rw] = self.VarMin + np.random.rand() * (self.VarMax - self.VarMin)
        return offspring
   
    def evaluation_and_update(self,  M, M_sort, j, offspring):        
        # Evaluate the offspring and update the population
        new_cost = self.CostFunction(offspring)
        self.FEs += 1
        flag = 0
        # Place the best offspring in the population
        if new_cost <= self.fitness[M[M_sort[j]]]:
            flag = 1
            self.fitness[M[M_sort[j]]] = new_cost
            self.pop[:, [M[M_sort[j]]]] = offspring
            if new_cost < self.the_best_cost:
              self.the_best_cost = new_cost[0]
              self.the_best_value = offspring
        else:
            self.Count += 1
        return flag
    
    def self_organizing(self, action):
        
        # Migrant selection: m
        M = np.random.choice(range(self.PopSize),self.m,replace=False)
        M_sort = np.argsort(self.fitness[M])
        for j in range(self.n):
            # Get the Migrant position (solution values) in the current population
            Migrant = self.pop[:, M[M_sort[j]]].reshape(self.dimension, 1)

            # Leader selection: k
            K = np.random.choice(range(self.PopSize),self.k,replace=False)
            K_sort = np.argsort(self.fitness[K])
            Leader = self.pop[:, K[K_sort[1]]].reshape(self.dimension, 1)
            if M[M_sort[j]] == K[K_sort[1]]:
                Leader = self.pop[:, K[K_sort[2]]].reshape(self.dimension, 1)

            # Perform self-organizing process on the Migrant  flag, move = 0, 1
            flag, move = 0, 1
            while (flag == 0) and (move <= 10):
                offspring = self.migrate_to_leader(Migrant, Leader, move, action)
                flag = self.evaluation_and_update(M, M_sort, j, offspring) 
                # Replace migrants if the optimization process is stuck
                move = move + 1
        self.migrants_replacement(action)
    
    def migrants_replacement(self, action):
        if self.Count > self.PopSize * 50:
          if self.the_best_cost == self.best_cost_old:
            rat = round(0.1 * self.PopSize)
            pop_temp = self.VarMin + np.random.rand(self.dimension, rat)*(self.VarMax-self.VarMin)
            fit_temp = self.CostFunction(pop_temp)
            self.FEs += rat
            D = np.random.choice(range(self.PopSize),rat,replace=False)
            self.pop[:,D] = pop_temp
            self.fitness[D] = fit_temp
            the_best_cost = min(self.fitness)
            if the_best_cost < self.the_best_cost:
              self.the_best_cost = the_best_cost
              self.the_best_value = self.pop[:, np.argmin(self.fitness)]
          else:
            self.best_cost_old = self.the_best_cost
          self.Count = 0

    def step(self, action):
        done = False
        #Error value smaller than 10**-8 will be taken as zero
        if(self.FEs > self.Max_FEs or
              abs(self.the_best_cost - self.f_global) < 10**-8
        ):
            print("--------------------------------------------------")
            print('Done function : ', self.function_name  , '  Best Cost : ', self.the_best_cost, '  FEs : ', self.FEs, '  Migration : ', self.Migration, ' Best Value : ', self.the_best_value[0] )
            print('Global Best Cost : ', self.f_global, '  Global Best Value : ', self.x_global)
            print("Difference : ", abs(self.f_global - self.the_best_cost), "  Percentage : ", (self.f_global - self.the_best_cost) / self.f_global * 100, "%")
            print("--------------------------------------------------")
            done = True
            #write the simple results to results.txt
            if self.write:
                filename = "results.txt"
                if self.pretrained:
                    filename = "pretrained_results.txt"
                with open(os.path.join(self.log_dir_results, filename), "a") as f:
                    f.write(self.function_name + "\t" + str(abs(self.f_global - self.the_best_cost)) + "\n")
        self.Migration = self.Migration + 1 
        self.total_migrations += 1

        if(len(self.history) > self.HistoryLen - 1):
            del self.history[0]            
            del self.difference_history[0]
            del self.action[0]

        before_FEs = self.FEs
        before_best_cost = self.the_best_cost
        for _ in range(25):
            self.self_organizing(action)

        self.history.append(self.the_best_cost)
        self.difference_history.append(abs(before_best_cost - self.the_best_cost))
        self.action.append(action[0])
        info = {}
        reward = self.get_reward(before_best_cost, before_FEs)
        observation = self.get_observation()

        #write action to file (name is function as a safe string to file name)
        if self.write:
            with open(os.path.join(self.log_dir_results,"action_" + "".join([c for c in self.function_name if re.match(r'\w', c)]) + ".txt"), "a") as f:
                f.write(str(self.total_migrations) + "\t" + str(self.prtFunc(action))  + "\t" + str(self.the_best_cost) + "\t" + str(self.FEs) + "\t" + str(reward) + "\n")
        return observation, reward, done, info

    def get_reward(self,before_best_cost, before_FEs):
        difference = abs(before_best_cost - self.the_best_cost)
        self.total_difference += difference
        reward = 0
        if difference > 0:
            reward = difference * (self.total_difference ** 2)
            if reward > self.total_difference:
                return self.total_difference
        reward = reward - ((self.FEs - before_FEs)  / (self.Max_FEs)** 2)
        return reward

    def get_observation(self):     
        history = np.array(self.history, dtype=np.float32)
        difference_history = np.array(self.difference_history, dtype=np.float32)
        action = np.array(self.action, dtype=np.float32)
        pad_value = self.history[0]  
        pad_difference_history = self.difference_history[0]  
        pad_action = self.action[0]  
        padded_history = np.pad(history, (self.HistoryLen - history.shape[0],0), mode='constant', constant_values=pad_value).reshape(self.HistoryLen,1)
        padded_difference_history = np.pad(difference_history, (self.HistoryLen - difference_history.shape[0],0), mode='constant', constant_values=pad_difference_history).reshape(self.HistoryLen,1)
        padded_action = np.pad(action, (self.HistoryLen - action.shape[0],0), mode='constant', constant_values=pad_action).reshape(self.HistoryLen,1)
        return {
            "history": padded_history,
            "difference_history": padded_difference_history,
            "action": padded_action,
        }

    def reset(self):
        self.the_best_value = None
        self.Count = 0
        self.Migration = 0

        index = np.random.randint(0, len(self.CostFunctions))
        selected = self.CostFunctions[index]
        self.CostFunction = selected[0]
        self.VarMin = -selected[1]
        self.VarMax = selected[1]

        self.pop = self.VarMin + np.random.rand(self.dimension, self.PopSize) * (self.VarMax - self.VarMin)
        self.fitness = self.CostFunction(self.pop)
        self.the_best_cost = min(self.fitness)        
        self.best_cost_old = self.the_best_cost
        
        self.history = [self.the_best_cost]
        self.difference_history = [self.the_best_cost]
        self.action = [0]
        self.FEs = self.PopSize
        print('Cost function : ', index)


        return self.get_observation()  # reward, done, info can't be included

    def render(self, mode="ansi"):
        print('Step at Migration :  ', self.Migration)
        print('The number of FEs :  ', self.FEs)
        print('The best cost     :  ', self.the_best_cost)
        print('Solution values   :  ', self.the_best_value)

    def close(self):
        pass
