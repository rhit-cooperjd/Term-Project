import numpy as np
import matplotlib.pyplot as plt

DEST_REACHED = 1
DEST_NOT_REACHED = 0

class Destination():
    def __init__(self, xpos, ypos, radius):
        self.xpos = xpos
        self.ypos = ypos
        self.radius = radius
        self.maxPoints = 10

    def destinationReached(self, agentX, agentY):
        self.distFromDest = np.sqrt((self.xpos - agentX)**2 + (self.ypos - agentY)**2)
        if(self.distFromDest < self.radius):
            return DEST_REACHED
        else:
            return DEST_NOT_REACHED

    


