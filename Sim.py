############################# B.C. (Before Claude) ###########################
# import Agent
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import fnn
# import Destination as dt
# import ea

# # lights = Agent.Lights()

# fig, ax = plt.subplots()
# # circle = ml.Circle((3, 2), 0.5, color='green', alpha = 0.3) 
# # circle2 = ml.Circle((2, 3.5), 0.5, color='green', alpha = 0.3) 
# def gradient_circle(ax, center, radius, color, steps):
#     for i in range(steps):
#         inner_radius = radius * (i / steps)
#         outer_radius = radius * ((i + 1) / steps)
#         alpha = 1 - (i / steps)  # Gradually decreasing opacity
#         circle = plt.Circle(center, outer_radius, color=color, alpha=alpha, fill=True)
#         ax.add_artist(circle)
# def draw_flag(ax, base_x, base_y, pole_height=1, flag_width=0.5, flag_height=0.4, color='blue'):
#     # Draw the pole as a vertical line
#     ax.plot([base_x, base_x], [base_y, base_y + pole_height], color='black', linewidth=2)
    
#     # Draw the flag as a triangle
#     flag = patches.Polygon(
#         [
#             (base_x, base_y + pole_height),                  # Bottom of the flag
#             (base_x + flag_width, base_y + pole_height - flag_height / 2),  # Top corner of the flag
#             (base_x, base_y + pole_height - flag_height)     # Bottom corner of the flag
#         ],
#         closed=True,
#         color=color
#     )
#     ax.add_patch(flag)

# flag = dt.Destination(100, 100, 3)

# layers = [2,4,2]

# # nn = fnn.FNN(layers)
# # paramnum = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
# # param_range = 1

# # Parameters of the evolutionary algorithm
# genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
# print("Number of parameters:",genesize)
# popsize = 50 
# recombProb = 0.5
# mutatProb = 0.01
# tournaments = 50*popsize 
# walker = Agent.Agent()

# xlist = []
# ylist = []
# distList = []

# def fitnessFunction(genotype):
#     nn = fnn.FNN(layers)
#     nn.setParams(genotype)
#     xlist, ylist, distlist, duration = sim(1000, nn)
#     distance = np.sqrt((walker.x - flag.xpos)**2 + (walker.y - flag.ypos)**2)
#     max_distance = np.sqrt(flag.xpos**2 + flag.ypos**2)
#     return np.clip(1 - (distance/max_distance), 0, 1)


# def sim(cycles, nn):
#     global xlist, ylist, distList
#     xlist = np.zeros(cycles + 1)
#     ylist = np.zeros(cycles + 1)
#     distList = np.zeros(cycles + 1)
#     duration = 0
#     destinationReached = walker.sense(flag)
#     for i in range(cycles):
#         if(destinationReached):
#             break
#     # while not destinationReached or not duration >= cycles:
#         # print(f"Destination Reached: {destinationReached}, Duration: {duration}")
#         walker.move(nn)
#         destinationReached = walker.sense(flag)
#         xlist[i+1] = walker.x
#         ylist[i+1] = walker.y
#         distList[i+1] = walker.distanceBetweenTwoPoints()
#         duration += 1
#     return xlist, ylist, distList, duration

# def show_last_gen():
#     global xlist, ylist
#     plt.plot(xlist[0], ylist[0],'ko')
#     plt.plot(flag.xpos, flag.ypos, 'ro')
#     plt.plot(xlist[-1], ylist[-1], 'go')
#     plt.show()

# ga = ea.MGA(fitnessFunction, genesize, popsize, recombProb, mutatProb, tournaments)
# ga.run()
# ga.showFitness()
# show_last_gen()

# # params = np.random.uniform(low=-param_range,high=param_range,size=paramnum)
# # nn.setParams(params)
# # for i in range(1):
# #     plt.plot(flag.xpos, flag.ypos, 'ro')
# #     xsim, ysim, dsim = sim(1000)
# #     # .set_aspect('equal')
# #     plt.plot(xsim, ysim)
# #     plt.plot(xsim[0], ysim[0], 'ko')
# #     plt.plot(xsim[-1], ysim[-1], 'go')
# #     plt.show()

# # #single agent plot


# # # Add a flag at (5,5)
# # draw_flag(ax, base_x=3, base_y=4, color='blue')

# # # Add gradient circles
# # gradient_circle(ax, (3, 2), 0.5, 'green', 5)  # Gradient at (5, 5)
# # gradient_circle(ax, (2, 3.5), 0.5, 'green', steps=5)   # Example gradient at (7, 8)

##################### A.C. (After Claude) #############################################
import numpy as np
import matplotlib.pyplot as plt
import fnn
import ea
import Agent
import Destination as dt

class ImprovedSimulation:
    def __init__(self):
        self.flag = dt.Destination(-100, -100, 3)
        self.layers = [2, 4, 2]
        
        # EA parameters
        self.genesize = np.sum(np.multiply(self.layers[1:], self.layers[:-1])) + np.sum(self.layers[1:])
        self.popsize = 50
        self.recombProb = 0.5
        self.mutatProb = 0.01
        self.tournaments = 50 * self.popsize

    def fitnessFunction(self, genotype):
        # Create fresh instances for each evaluation
        walker = Agent.Agent()
        nn = fnn.FNN(self.layers)
        nn.setParams(genotype)
        
        # Run simulation
        total_fitness = 0
        prev_distance = np.sqrt(self.flag.xpos**2 + self.flag.ypos**2)  # Initial distance
        steps_without_improvement = 0
        max_steps_without_improvement = 100
        
        for step in range(1000):
            # Move agent
            walker.move(nn)
            reached = walker.sense(self.flag)
            
            # Calculate current distance to goal
            current_distance = np.sqrt((walker.x - self.flag.xpos)**2 + 
                                    (walker.y - self.flag.ypos)**2)
            
            # Reward for getting closer to target
            if current_distance < prev_distance:
                total_fitness += (prev_distance - current_distance) / prev_distance
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            # Early termination conditions
            if reached:
                total_fitness += 100  # Big bonus for reaching target
                break
            if steps_without_improvement >= max_steps_without_improvement:
                break
                
            prev_distance = current_distance
        
        # Penalize final distance
        max_possible_distance = np.sqrt(self.flag.xpos**2 + self.flag.ypos**2)
        distance_penalty = current_distance / max_possible_distance
        
        final_fitness = total_fitness * (1 - distance_penalty)
        return np.clip(final_fitness, 0, 100)

    def run_evolution(self):
        ga = ea.MGA(self.fitnessFunction, self.genesize, self.popsize, 
                    self.recombProb, self.mutatProb, self.tournaments)
        ga.run()
        return ga

    def visualize_best_agent(self, ga):
        # Get best genotype
        best_idx = int(ga.bestind[-1])
        best_genotype = ga.pop[best_idx]
        
        # Create neural network with best weights
        nn = fnn.FNN(self.layers)
        nn.setParams(best_genotype)
        
        # Run simulation with best agent
        walker = Agent.Agent()
        positions = []
        
        for _ in range(1000):
            positions.append((walker.x, walker.y))
            walker.move(nn)
            if walker.sense(self.flag):
                break
                
        positions = np.array(positions)
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Path')
        plt.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
        plt.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
        plt.plot(self.flag.xpos, self.flag.ypos, 'k*', label='Goal')
        plt.legend()
        plt.grid(True)
        plt.show()

# Usage:
sim = ImprovedSimulation()
ga = sim.run_evolution()
ga.showFitness()
sim.visualize_best_agent(ga)