import numpy as np
from typing import List, Tuple
import fnn
import ea
import Destination as dt
import moea
import Agent

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
        self.agent_paths = [[] for _ in range(5000)] # num_generations=100
        self.cycles = self.num_generations,
        self.final_checkpoint_score = [[] for _ in range(5000)]
        self.final_path_score = [[] for _ in range(5000)]
        self.weights = [[] for _ in range(5000)]
        self.fitnesses = [[] for _ in range(5000)]
        
    def train_hybrid_system(self):
        paths, objectives = self.path_planner.optimize()
        
        # Select best path using Chebyshev decomposition
        best_scalar = float('-inf')
        best_path = None
        best_weight = None
        
        for weight in self.path_planner.weights:
            for path, obj in zip(paths, objectives):
                scalar = self._chebyshev_scalar(obj, weight)
                if scalar > best_scalar:
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
        
        current_path = []

        path_length = 0
        best_checkpoint_distances = {i: float('inf') for i in range(len(self.checkpoints))}
        
        prev_pos = np.array([0, 0])
        
        for step in range(duration):
            current_pos = np.array([agent.x, agent.y])

            # if duration % 100 == 0:
            #     current_path.append((agent.x, agent.y))

            current_path.append((agent.x, agent.y))
            
            # Update path length
            step_distance = np.linalg.norm(current_pos - prev_pos)
            path_length += step_distance
            
            # Update checkpoint distances
            for i, checkpoint in enumerate(self.checkpoints):
                dist = np.linalg.norm(current_pos - checkpoint)
                best_checkpoint_distances[i] = min(best_checkpoint_distances[i], dist)
            
            if agent.sense(self.flag) == dt.DEST_REACHED:
                break
            
            # Path following calculations
            path_dists = [np.linalg.norm(current_pos - point) for point in target_path]
            closest_idx = np.argmin(path_dists)
            path_error = min(path_dists)
            
            next_idx = min(closest_idx + 1, len(target_path) - 1)
            target_vector = target_path[next_idx] - current_pos
            path_angle = np.arctan2(target_vector[1], target_vector[0]) - agent.direction
            
            # Calculate checkpoint-related inputs
            checkpoint_inputs = []
            for checkpoint in self.checkpoints:
                checkpoint_pos = np.array(checkpoint)
                # Distance to checkpoint
                dist_to_checkpoint = np.linalg.norm(current_pos - checkpoint_pos)
                # Angle to checkpoint
                to_checkpoint = checkpoint_pos - current_pos
                angle_to_checkpoint = np.arctan2(to_checkpoint[1], to_checkpoint[0]) - agent.direction
                checkpoint_inputs.extend([dist_to_checkpoint, angle_to_checkpoint])
            
            # Combine all neural network inputs
            nn_inputs = np.array([
                agent.leftSensor,
                agent.rightSensor,
                path_error,
                path_angle,
                *checkpoint_inputs
            ])
            
            agent.move(self.nn, nn_inputs)
            prev_pos = current_pos
            
            if step == 999:
                path_length += 1000
        
        if agent.sense(self.flag) == dt.DEST_REACHED:
            final_dist = np.linalg.norm(current_pos - np.array([self.flag.xpos, self.flag.ypos]))
            if final_dist < self.flag.radius:
                path_length -= (self.flag.radius - final_dist)
        
        checkpoint_score = 0
        for dist in best_checkpoint_distances.values():
            checkpoint_score += 1 / (1 + dist) * 10
        
        # if current_path:
        #     self.agent_paths.append(current_path)
        #     self.agent_path_index += 1

        self.agent_paths.append(current_path)
        self.weights.append(weight)
        self.agent_path_index += 1
        
        norm_path_length = path_length / self.path_planner.reference_point[0]
        norm_checkpoint_score = checkpoint_score / len(self.checkpoints)

        fitness = (weight[0] * (1 - norm_path_length) + weight[1] * norm_checkpoint_score)

        self.fitnesses.append(fitness)
        print(fitness)

        return fitness