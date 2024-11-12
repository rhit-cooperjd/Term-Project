import hybrid_controller as controller
import Destination as dt
import matplotlib.pyplot as plt
    
checkpoints = [(10, 20), (30, 45), (50, 50)]

layers = [4, 4, 2]

flag = dt.Destination(100, 100, 3)

sim = controller.HybridController(
    checkpoints=checkpoints,
    layers=layers,
    flag=flag
)

best_path, best_ind, best_weight = sim.train_hybrid_system()

plt.plot(flag.xpos, flag.ypos, 'r*', label="Destination")
checkpoints_x = [c[0] for c in checkpoints]
checkpoints_y = [c[1] for c in checkpoints]
plt.plot(checkpoints_x, checkpoints_y, 'gx', label="Checkpoints")

# plt.plot(best_path, label="best path")

best_path_x = [point[0] for point in best_path]
best_path_y = [point[1] for point in best_path]
plt.plot(best_path_x, best_path_y, label="Best path")

for i in range(sim.agent_path_index-1):
    # plt.plot(sim.agent_paths[i])
    path_x = [point[0] for point in sim.agent_paths[i]]
    path_y = [point[1] for point in sim.agent_paths[i]]
    plt.plot(path_x, path_y, alpha=0.3)

last_path_x = [point[0] for point in sim.agent_paths[sim.agent_path_index]]
last_path_y = [point[1] for point in sim.agent_paths[sim.agent_path_index]]
plt.plot(last_path_x, last_path_y, label="last gen")

plt.legend()
plt.show()
print(best_weight)

