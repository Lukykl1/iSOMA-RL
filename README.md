### README for Reinforcement Learning Script

#### Description
This script implements a reinforcement learning model using Stable Baselines3, a popular reinforcement learning library. It's designed to work with custom environments and utilizes various functionalities like monitoring, normalization, and policy learning. The script's primary function is to train a Proximal Policy Optimization (PPO) model on different benchmark functions from the `opfunu` library, which is commonly used for optimization problems.

#### Requirements
- Python 3.x
- Gym
- NumPy
- Matplotlib
- Stable Baselines3
- opfunu library

#### Installation
Describe the steps for installing necessary libraries, e.g.:
```
pip install gym numpy matplotlib stable-baselines3 opfunu timeit tqdm
```

#### CustomEnv
CustomEnv is a key component of this script. It is a custom implementation that combines the iSOMA algorithm with Reinforcement Learning using PPO. This environment is designed to simulate optimization problems, making it an excellent tool for testing and evaluating RL-based optimization strategies. CustomEnv can be used as a drop-in replacement in various scenarios where iSOMA-like behavior is desired in a reinforcement learning context.

#### Variants of iSOMA Parameter Optimization Using PPO RL
- **mnk_isoma_rl**: This variant focuses on optimizing the 'm', 'n', and 'k' parameters of iSOMA using Proximal Policy Optimization (PPO) Reinforcement Learning (RL).
- **prt_isoma_rl**: This variant is designed to optimize the 'prt' parameter of iSOMA employing the PPO RL approach.
- **step_isoma_rl**: In this variant, the focus is on optimizing both the 'step' and 'step size' parameters of iSOMA, utilizing PPO RL methodology.


#### Script Arguments
The script takes several command-line arguments to customize its behavior:

1. **sys.argv[1]**: Identifier for saving models and logs.
2. **sys.argv[2]**: Number of pretraining. Not used in article iSOMA-RL
3. **sys.argv[3]**: Year of the `opfunu` function class (e.g., '2013', '2015').
4. **sys.argv[4]**: Step size for iterating through `opfunu` functions.
5. **sys.argv[5]**: Start index for `opfunu` functions.
6. **sys.argv[6]**: Number of iterations to run each function.
7. **sys.argv[7]**: End index for `opfunu` functions.
8. **sys.argv[8]**: Directory for saving results.

#### How to Run
Example command to run the script:
```
python script_name.py [arg1] [arg2] [arg3] [arg4] [arg5] [arg6] [arg7] [arg8]
```

#### Script Workflow
1. **Environment Setup**: Creates a custom environment based on parameters and `opfunu` functions.
2. **Model Training**: Trains a PPO model on the custom environment.
3. **Evaluation**: Evaluates the model on different `opfunu` functions.
4. **Logging and Results**: Saves logs and performance results to the specified directory.

#### Output
The script saves the trained model and logs the performance metrics in the specified directory.
