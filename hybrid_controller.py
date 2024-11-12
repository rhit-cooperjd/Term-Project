import numpy as np
from typing import List, Tuple
import fnn
import ea
import Destination as dt
import moea
import Agent
import Destination as dt

class HybridController:
    def __init__(self, checkpoints: List[Tuple[float, float]], layers, flag):
        self.checkpoints = checkpoints
        self.flag = flag
        self.num_generations = 100
        self.path_planner = moea.MOEA(
            start=(0, 0),
            end=(flag.xpos, flag.ypos),
            checkpoints=checkpoints,
            destination_radius=flag.radius,
            population_size=50,
            num_generations=self.num_generations,
            num_weight_vectors=5
        )
        self.agent_path_index = 0
        self.layers = layers
        self.nn = fnn.FNN(self.layers)
        self.agent_paths = [[] for _ in range(100)] # num_generations=100
        self.cycles = self.num_generations
        
    def train_hybrid_system(self):
        paths, objectives = self.path_planner.optimize()
        
        # Select best path using Chebyshev decomposition
        best_scalar = float('inf')
        best_path = None
        best_weight = None
        
        for weight in self.path_planner.weights:
            for path, obj in zip(paths, objectives):
                scalar = self._chebyshev_scalar(obj, weight)
                if scalar < best_scalar:
                    best_scalar = scalar
                    best_path = path
                    best_weight = weight
        
        def fitness_function(genotype):
            self.nn.setParams(genotype)
            return self.evaluate_path_following(best_path, best_weight)
            
        genesize = np.sum(np.multiply(self.layers[1:], self.layers[:-1])) + np.sum(self.layers[1:])
        ga = ea.MGA(fitness_function, genesize, popsize=50, recomprob=0.5, mutationprob=0.01, tournaments=2500)
        ga.run()
        ga.showFitness()
        
        return best_path, ga.pop[int(ga.bestind[-1])], best_weight
        
    def _chebyshev_scalar(self, objectives: Tuple[float, float], weight: np.ndarray) -> float:
        """Calculate Chebyshev scalarization value"""
        norm_dist = objectives[0] / self.path_planner.reference_point[0]
        norm_checkpoints = objectives[1] / self.path_planner.reference_point[1]
        return max(weight[0] * norm_dist, weight[1] * norm_checkpoints)
        
    def evaluate_path_following(self, target_path: np.ndarray, weight: np.ndarray) -> float:
        """Evaluate using both path following and checkpoint coverage"""
        agent = Agent.Agent()
        duration = 1000
        if duration % self.num_generations == 0:
            current_path = [(0, 0)]
        path_error = 0
        checkpoint_error = 0
        
        # Track minimum distances to checkpoints
        min_checkpoint_distances = {i: float('inf') for i in range(len(self.checkpoints))}
        
        for step in range(duration):
            if agent.sense(self.flag) == dt.DEST_REACHED:
                path_error -= 100  # Reward for reaching destination
                break
                
            # Calculate path following error
            path_dists = [np.linalg.norm(np.array([agent.x, agent.y]) - point) for point in target_path]
            path_error += min(path_dists)
            
            # Update minimum distances to checkpoints
            for i, checkpoint in enumerate(self.checkpoints):
                dist = np.linalg.norm(np.array([agent.x, agent.y]) - checkpoint)
                min_checkpoint_distances[i] = min(min_checkpoint_distances[i], dist)
            
            # Move agent
            next_idx = np.argmin(path_dists) + 1
            if next_idx < len(target_path):
                target_vector = target_path[next_idx] - np.array([agent.x, agent.y])
                path_angle = np.arctan2(target_vector[1], target_vector[0]) - agent.direction
                
                nn_inputs = np.array([
                    agent.leftSensor,
                    agent.rightSensor,
                    min(path_dists),  # Current path error
                    path_angle
                ])

                
                agent.move(self.nn, nn_inputs)
                if self.cycles % self.num_generations == 0:
                    current_path.append((agent.x, agent.y))
                
            
            if step == 999:
                path_error += 500  # Timeout penalty
        
        if self.cycles % 10 == 0:
            print("index" + str(self.agent_path_index))
            self.agent_paths[self.agent_path_index] = current_path
            self.agent_path_index += 1
        # Calculate checkpoint coverage error
        checkpoint_error = sum(min_checkpoint_distances.values())
        
        # Normalize objectives
        max_path_error = 1000  # Maximum possible path error (including timeout)
        max_checkpoint_error = np.sqrt(2) * 100 * len(self.checkpoints)  # Maximum possible checkpoint distance
        
        norm_path_error = path_error / max_path_error
        norm_checkpoint_error = checkpoint_error / max_checkpoint_error
        
        # Calculate Chebyshev scalar for fitness
        scalar = max(weight[0] * norm_path_error, weight[1] * norm_checkpoint_error)
        
        return 1.0 / (1.0 + scalar)  # Convert to fitness score (0 to 1)