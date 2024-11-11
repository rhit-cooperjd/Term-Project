##################################################################################
# Example script for the forward neural network
# A random neural network with 2 inputs and 1 output
#
# Eduardo Izquierdo
# September 2024
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import fnn 

layers = [2,10,2]

a = fnn.FNN(layers)

# number of weights + number of biases 
paramnum = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
range = 1 
params = np.random.uniform(low=-range,high=range,size=paramnum)
a.setParams(params)

x = np.arange(-10, 10.5, 0.1)
y = np.arange(-10, 10.5, 0.1)
z1 = np.zeros((len(x),len(y)))
z2 = np.zeros((len(x),len(y)))

ii = 0
for i in x:
    ji = 0
    for j in y:
        outputs = a.forward([i,j])
        z1[ii][ji] =outputs[0][0]
        z2[ii][ji] =outputs[0][1]
        ji += 1
    ii += 1

plt.pcolormesh(x,y,z1,cmap='jet')
plt.colorbar()
plt.show()
plt.pcolormesh(x,y,z2,cmap='jet')
plt.colorbar()
plt.show()