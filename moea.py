import numpy as np
from typing import List, Tuple
import random

class MOEA:
    def __init__(
        self, 
        start: Tuple[float, float],
        end: Tuple[float, float],
        checkpoints: List[Tuple[float, float]],
        destination_radius: float,
        population_size: int = 100,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        num_weight_vectors: int = 10
    ):
        self.start = np.array(start)
        self.end = np.array(end)
        self.destination_radius = destination_radius
        self.checkpoints = [np.array(cp) for cp in checkpoints]
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_weight_vectors = num_weight_vectors
        self.max_checkpoint_point_value = 50
        
        # Generate uniformly distributed weight vectors for Chebyshev decomposition
        self.weights = self._generate_weight_vectors()
        
        # Initialize reference point for objective normalization
        # self.reference_point = np.array([float('inf'), float('inf')])
        max_path_length = np.linalg.norm(self.end-self.start) * 2
        max_checkpoint_score = len(self.checkpoints) * self.max_checkpoint_point_value
        self.reference_point = np.array([max_path_length, max_checkpoint_score])

    def _generate_weight_vectors(self) -> np.ndarray:
        """Generate weight vectors for Chebyshev decomposition"""
        weights = []
        for i in range(self.num_weight_vectors):
            w = i / (self.num_weight_vectors - 1)
            weights.append([w, 1 - w])
        return np.array(weights)
    
    def _initialize_population(self) -> List[np.ndarray]:
        """Initialize population with random waypoints between start and end"""
        population = []
        for _ in range(self.population_size):
            # Variable number of waypoints
            num_waypoints = random.randint(2, max(3, len(self.checkpoints)))
            path = [self.start]
            
            # Add waypoints considering checkpoint locations
            for _ in range(num_waypoints):
                if random.random() < 0.5 and self.checkpoints:  # 50% chance to use checkpoint as guide
                    checkpoint = random.choice(self.checkpoints)
                    noise = np.random.normal(0, 5, 2)  # Add some noise around checkpoint
                    waypoint = checkpoint + noise
                else:
                    # Random point between start and end
                    x = random.uniform(min(self.start[0], self.end[0]), max(self.start[0], self.end[0]))
                    y = random.uniform(min(self.start[1], self.end[1]), max(self.start[1], self.end[1]))
                    waypoint = np.array([x, y])
                path.append(waypoint)
                
            path.append(self.end)
            population.append(np.array(path))
        return population
    
    def _evaluate_objectives(self, path: np.ndarray) -> Tuple[float, float]:
        """Calculate path length and checkpoint coverage objectives"""
        # Calculate path length up to destination radius
        distance = 0
        for i in range(len(path) - 1):
            if i == len(path) - 2:
                # For last segment, consider destination radius
                remaining = path[i+1] - path[i]
                remaining_dist = np.linalg.norm(remaining)
                if remaining_dist > self.destination_radius:
                    distance += remaining_dist - self.destination_radius
            else:
                distance += np.linalg.norm(path[i+1] - path[i])

        checkpoint_score = 0
        for checkpoint in self.checkpoints:
            min_distance = float('inf')
            for point in path:
                dist = np.linalg.norm(point - checkpoint)
                min_distance = min(min_distance, dist)
            checkpoint_score = 1 / (1 + min_distance)

        return distance, checkpoint_score
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Intelligent crossover considering path structure"""
        if len(parent1) != len(parent2):
            # If different lengths, interpolate to get same number of points
            new_length = max(3, min(len(parent1), len(parent2)))
            child = []
            child.append(self.start)  # Always keep start point
            
            # Interpolate middle points
            for i in range(1, new_length-1):
                t = i / (new_length - 1)
                if random.random() < 0.5:
                    idx1 = int(t * (len(parent1)-2)) + 1
                    point = parent1[idx1]
                else:
                    idx2 = int(t * (len(parent2)-2)) + 1
                    point = parent2[idx2]
                child.append(point)
                
            child.append(self.end)  # Always keep end point
            return np.array(child)
        else:
            # If same length, do point-wise crossover
            child = []
            child.append(self.start)  # Keep start point
            
            # Crossover middle points
            for i in range(1, len(parent1)-1):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
                    
            child.append(self.end)  # Keep end point
            return np.array(child)
    
    def _mutate(self, path: np.ndarray) -> np.ndarray:
        """Mutation with adaptive step size"""
        mutated_path = path.copy()
        
        # Don't mutate start and end points
        for i in range(1, len(path) - 1):
            if random.random() < self.mutation_rate:
                # Adaptive mutation: larger steps when far from checkpoints
                min_checkpoint_dist = float('inf')
                for checkpoint in self.checkpoints:
                    dist = np.linalg.norm(path[i] - checkpoint)
                    min_checkpoint_dist = min(min_checkpoint_dist, dist)
                
                # Scale mutation size based on distance to nearest checkpoint
                mutation_scale = min(1.0, min_checkpoint_dist / 10.0)
                offset = np.random.normal(0, mutation_scale, 2)
                mutated_path[i] += offset
                
        return mutated_path
    
    def optimize(self) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """Run the multi-objective optimization"""
        # Initialize population
        population = self._initialize_population()
        
        # Main optimization loop
        for generation in range(self.num_generations):
            # Evaluate all solutions
            objectives = [self._evaluate_objectives(path) for path in population]
            
            # Update reference point for normalization
            self.reference_point = np.minimum(self.reference_point, 
                                           np.min(objectives, axis=0))
            
            # Create new population
            new_population = []
            
            for _ in range(self.population_size):
                # Tournament selection using random weight vector
                weight = random.choice(self.weights)
                tournament_size = 3
                tournament = random.sample(list(enumerate(population)), tournament_size)
                
                best_idx = None
                best_score = float('inf')
                
                for idx, solution in tournament:
                    score = max(weight[0] * (objectives[idx][0] / self.reference_point[0]),
                              weight[1] * (objectives[idx][1] / self.reference_point[1]))
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                
                # Create offspring
                parent1 = population[best_idx]
                parent2 = population[random.randint(0, self.population_size-1)]
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Get final objectives
        final_objectives = [self._evaluate_objectives(path) for path in population]
        
        # Filter for Pareto front
        pareto_indices = []
        for i, obj1 in enumerate(final_objectives):
            is_dominated = False
            for j, obj2 in enumerate(final_objectives):
                if i != j:
                    if (obj2[0] <= obj1[0] and obj2[1] >= obj1[1] and 
                        (obj2[0] < obj1[0] or obj2[1] > obj1[1])):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_indices.append(i)
        
        pareto_paths = [population[i] for i in pareto_indices]
        pareto_objectives = [final_objectives[i] for i in pareto_indices]
        
        return pareto_paths, pareto_objectives