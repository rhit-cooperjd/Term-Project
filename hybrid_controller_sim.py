import hybrid_controller as controller
import Destination as dt
import matplotlib.pyplot as plt
import numpy as np
    
checkpoints = [(10, 20), (30, 45), (50, 60)]

layers = [4 + 2*len(checkpoints), 4, 2]

flag = dt.Destination(100, 100, 3)

sim = controller.HybridController(
    checkpoints=checkpoints,
    layers=layers,
    flag=flag
)

def find_index_of_best_ind(sim):
    best_fitness = 0
    best_index = None
    for i, f in enumerate(sim.fitnesses):
        if f > best_fitness:
            best_fitness = f
            best_index = i
    return best_index

best_path, best_ind, best_weight = sim.train_hybrid_system()

plt.plot(flag.xpos, flag.ypos, 'r*', label="Destination")
checkpoints_x = [c[0] for c in checkpoints]
checkpoints_y = [c[1] for c in checkpoints]
plt.plot(checkpoints_x, checkpoints_y, 'gx', label="Checkpoints")

# plt.plot(best_path, label="best path")

# best_path_x = [point[0] for point in best_path]
# best_path_y = [point[1] for point in best_path]
# plt.plot(best_path_x, best_path_y, label="Best path")

# for i in range(sim.agent_path_index):
#     # plt.plot(sim.agent_paths[i])
#     path_x = [point[0] for point in sim.agent_paths[i]]
#     path_y = [point[1] for point in sim.agent_paths[i]]
#     plt.plot(path_x, path_y, alpha=0.3)

# print(sim.fitnesses)
highest_fitness_path_index = find_index_of_best_ind(sim)
highest_fitness_path_x = [point[0] for point in sim.agent_paths[highest_fitness_path_index]]
highest_fitness_path_y = [point[1] for point in sim.agent_paths[highest_fitness_path_index]]
plt.plot(highest_fitness_path_x, highest_fitness_path_y, color='k', label='Most Fit Agent')
plt.plot(f"highest fitness: {sim.agent_paths[highest_fitness_path_index]}")

# last_path_x = [point[0] for point in sim.agent_paths[sim.agent_path_index]]
# last_path_y = [point[1] for point in sim.agent_paths[sim.agent_path_index]]
# plt.plot(last_path_x, last_path_y, label="last gen")

plt.legend()
plt.show()
print(best_weight)

