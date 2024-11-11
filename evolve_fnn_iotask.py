##################################################################################
# Example script for evolving a feedforward neural network to solve XOR problem 
#
# Eduardo Izquierdo
# September 2024
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import fnn 
import ea

# Parameters of the XOR task
dataset = [[-1,-1],[-1,1],[1,-1],[1,1]]
labels = [0,1,1,0]

# Parameters for another task
# dataset = [[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1],[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]]
# labels = [1,1,1,1,1,1,1,1,0,0,0,0]

# Parameters of the neural network
layers = [2,3,1]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.5
mutatProb = 0.01
tournaments = 100*popsize 

def fitnessFunction(genotype):
    # Step 1: Create the neural network.
    a = fnn.FNN(layers)

    # Step 2. Set the parameters of the neural network according to the genotype.
    a.setParams(genotype)
    
    # Step 3. For each training point in the dataset, evaluate the current neural network.
    error = 0.0
    for i in range(len(dataset)):
        error += np.abs(a.forward(dataset[i]) - labels[i])

    return 1 - (error/len(dataset))

# Evolve
ga = ea.MGA(fitnessFunction, genesize, popsize, recombProb, mutatProb, tournaments)
ga.run()
ga.showFitness()

# Obtain best agent
best = int(ga.bestind[-1])
print(best)
a = fnn.FNN(layers)
a.setParams(ga.pop[best])

# Function to visualize the best solution
def viz(neuralnet, dataset, label):
    X = np.linspace(-1.05, 1.05, 100)
    Y = np.linspace(-1.05, 1.05, 100)
    output = np.zeros((100,100))
    i = 0
    for x in X: 
        j = 0
        for y in Y: 
            output[i,j] = neuralnet.forward([x,y])
            j += 1
        i += 1
    plt.contourf(X,Y,output)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(len(dataset)):
        if label[i] == 1:
            plt.plot(dataset[i][0],dataset[i][1],'wo')
        else:
            plt.plot(dataset[i][0],dataset[i][1],'wx')
    plt.show()  

# Visualize data
viz(a, dataset, labels)

