#import plot
import matplotlib.pyplot as plt
import numpy as np
import sys
#function to parse f.write(str(self.Migration) + "\t" + str(action[0]) + "\t" + str(action[1])  + "\t" + str(action[2])  + "\t" + str(self.the_best_cost) + "\t" + str(self.FEs) + "\t" + str(reward) + "\n")
def parse_file(filename):
    migrations = []
    actions = [[],[]]
    costs = []
    fes = []
    rewards = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split("\t")
            migrations.append(int(parts[0]))
            actions[0].append(float(parts[1]))
            actions[1].append(float(parts[2]))
            costs.append(float(parts[3]))
            fes.append(int(parts[4]))
            rewards.append(float(parts[5]))
    return migrations, actions, costs, fes, rewards

#plot the results
def plot_results(migrations, costs, fes, actions, rewards, axs):
    #plot migrations vs costs
    axs[0,0].plot(fes, costs)
    axs[0,0].set_xlabel('Migration')
    axs[0,0].set_ylabel('Cost')

    if(len(fes) > 3000):
        axs[1,1].plot(np.convolve(fes[3000:], np.ones((100,))/100, mode='valid'), np.convolve(costs[3000:], np.ones((100,))/100, mode='valid'))
        axs[1,1].set_xlabel('Migration')
        axs[1,1].set_ylabel('Cost')

    
    axs[0,1].plot(np.convolve(fes, np.ones((100,))/100, mode='valid'), np.convolve(actions[0], np.ones((100,))/100, mode='valid'))
    axs[0,1].set_xlabel('Migration')
    axs[0,1].set_ylabel('Move')
    
    axs[1,0].plot(np.convolve(fes, np.ones((100,))/100, mode='valid'), np.convolve(actions[1], np.ones((100,))/100, mode='valid'))
    axs[1,0].set_xlabel('Migration')
    axs[1,0].set_ylabel('Step')

    axs[2,0].plot(np.convolve(fes, np.ones((100,))/100, mode='valid'), np.convolve(rewards, np.ones((100,))/100, mode='valid'))
    axs[2,0].set_xlabel('Migration')
    axs[2,0].set_ylabel('Reward')
        
    axs[2,1].clear()
    axs[2,1].text(0.15,0.5, str(-500 - costs[-1])+ " / " + str(costs[-1]))
    axs[2,1].text(0.25,0.25, fes[-1])
    axs[2,1].text(0.25,0.75, rewards[-1])
    axs[2,1].text(0.25,0, fes[-1])
    plt.draw()
    fig.canvas.flush_events()

def show(filename, axs):
    migrations, actions, costs, fes, rewards = parse_file(filename)
    plot_results(migrations, costs, fes, actions, rewards, axs)

if __name__ == "__main__":  
    plt.ion()
    fig, axs = plt.subplots(3, 2)
    while True:
        show(sys.argv[1], axs)
        #sleep
        plt.pause(3)