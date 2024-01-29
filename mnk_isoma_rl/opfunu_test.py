"""
This script implements reinforcement learning using Stable Baselines3 library.
It is designed to train and evaluate a model on various optimization functions.
"""


import gym
import numpy as np
from gym.spaces import Dict, Discrete, Box
import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from env import CustomEnv
import stable_baselines3 as sb
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from functions import *
import opfunu
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.callbacks import BaseCallback
import timeit
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HistoryLen = 25

log_dir_results = "./" + sys.argv[8]


def make_env_pre():
    funcTypex = opfunu.get_functions_based_classname("2015")[
        np.random.randint(0, len(opfunu.get_functions_based_classname("2015")))
    ]
    funcx = funcTypex(ndim=10)
    # return dummy environment
    return CustomEnv(
        dimension=10,
        PopSize=100,
        Max_FEs=10 * 10**4,
        m=10,
        n=5,
        k=15,
        HistoryLen=HistoryLen,
        CostFunctions=[(lambda x: Wrapper(x, funcx.evaluate), funcx.bounds[0][1])],
        f_global=funcx.f_global,
        x_global=funcx.x_global,
        function_name=funcTypex.name,
        write=False,
        pretrained=False,
        log_dir_results=log_dir_results,
    )


callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)
env = DummyVecEnv([make_env_pre])
if (int(sys.argv[2])) > 0:
    env = DummyVecEnv([make_env_pre for _ in range(int(sys.argv[2]))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    log_dir = "/tmp/gym/" + str(sys.argv[1]) + "/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

model = PPO("MultiInputPolicy", env, verbose=0)

if (int(sys.argv[2])) > 0:
    model.learn(total_timesteps=650000, callback=callback_max_episodes, use_sde=True)

model.save("model" + str(sys.argv[1]) + ".bin")
# remove all *.txt files in the folder log
os.makedirs(log_dir_results, exist_ok=True)
filelist = [
    f
    for f in os.listdir(log_dir_results)
    if f.endswith(".txt") and not f.endswith("results.txt")
]
for f in filelist:
    os.remove(os.path.join(log_dir_results, f))

# timeit
start = timeit.default_timer()
for funcType in opfunu.get_functions_based_classname(sys.argv[3])[
    int(sys.argv[5]) : int(sys.argv[7]) : int(sys.argv[4])
]:
    if sys.argv[3] == "2013":
        name = ""
    else:
        name = sys.argv[3]
    for _ in range(int(sys.argv[6])):
        func = funcType(ndim=10)

        def make_env():
            return CustomEnv(
                dimension=10,
                PopSize=100,
                Max_FEs=10 * 10**4,
                m=10,
                n=5,
                k=15,
                HistoryLen=HistoryLen,
                CostFunctions=[
                    (lambda x: Wrapper(x, func.evaluate), func.bounds[0][1])
                ],
                f_global=func.f_global,
                x_global=func.x_global,
                function_name=sys.argv[3] + funcType.name,
                write=True,
                pretrained=False,
                log_dir_results=log_dir_results,
            )

        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1, verbose=1)
        print("---------------------------------------")
        print(funcType.name)

        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
        log_dir = "/tmp/gym/" + str(sys.argv[1]) + "/"
        os.makedirs(log_dir, exist_ok=True)

        env = VecMonitor(env, log_dir)
        model = model.load("model" + str(sys.argv[1]) + ".bin", env)
        model.set_env(env)
        model.learn(total_timesteps=55000, callback=callback_max_episodes)

# timeit end
stop = timeit.default_timer()
print("Time: ", stop - start)
