import os
import time
import random
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import tqdm
from datetime import datetime

# ==========================
# Utility Functions
# ==========================

def timestamp():
    """Returns a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_results_to_file(results_df, dataset_name):
    """Saves results to a timestamped CSV file."""
    filename = f"{dataset_name}_results_{timestamp()}.csv"
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def save_best_result(dataset_name, generations, best_route, avg_distances, best_distance):
    """Saves the best results to a separate file."""
    filename = f"best_{dataset_name}_{timestamp()}.txt"
    with open(filename, 'w') as file:
        file.write(f"Generations: {generations}\n")
        file.write(f"Best Distance: {best_distance}\n")
        file.write(f"Best Route: {best_route}\n")
        file.write(f"Avg Distances: {list(map(float, avg_distances))}\n")
    print(f"Best results saved to {filename}")

# ==========================
# TSP Data Processing
# ==========================

def prepare_tsp_files(filepath):
    """Parses a TSPLIB format file and extracts city coordinates."""
    print(f"Loading dataset: {filepath}")
    with open(filepath, 'r') as file:
        lines = file.readlines()
        coords = {}
        node_section = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                node_section = True
                continue
            if line.strip() == "EOF":
                break
            if node_section:
                parts = line.strip().split()
                node_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                coords[node_id] = (x, y)
    print(f"Loaded {len(coords)} cities from {filepath}")
    return coords

def compute_distance_matrix(coords):
    """Computes a distance matrix for TSP."""
    print("Computing distance matrix...")
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = euclidean(coords[i+1], coords[j+1])
    print("Distance matrix computed.")
    return distance_matrix

# ==========================
# Genetic Algorithm Components
# ==========================

def tournament_selection(population, fitness_values, tournament_size=2):
    """Selects an individual from the population using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    return population[tournament_indices[tournament_fitness.index(min(tournament_fitness))]]

def ordered_crossover(parent1, parent2):
    """Performs Ordered Crossover (OX1) on two parent routes."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    remaining = [item for item in parent2 if item not in child]
    index = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[index]
            index += 1
    return child

def pmx_crossover(parent1, parent2):
    """Performs Partially Mapped Crossover (PMX) on two parent routes."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}
    for i in range(size):
        if i < start or i >= end:
            value = parent2[i]
            while value in mapping:
                value = mapping[value]
            child[i] = value
    return child

def swap_mutation(route, mutation_rate):
    """Mutates a route using swap mutation."""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def scramble_mutation(route, mutation_rate):
    """Mutates a route using scramble mutation."""
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(route)), 2))
        subset = route[start:end]
        random.shuffle(subset)
        route[start:end] = subset
    return route

def calculate_fitness(route, distance_matrix):
    """Calculates the total distance of a route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i+1]]
    total_distance += distance_matrix[route[-1]][route[0]]
    return total_distance

def genetic_algorithm(distance_matrix, population_size, generations, mutation_rate, crossover_rate, crossover, mutation, tournament_size):
    """Executes the genetic algorithm with stopping criteria."""
    print(f"Running Genetic Algorithm with Population Size: {population_size}, Generations: {generations}")
    start_time = time.time()
    population = [random.sample(range(len(distance_matrix)), len(distance_matrix)) for _ in range(population_size)]
    avg_distances = []
    best_fitness = float('inf')
    # stagnant_generations = 0
    
    for generation in range(generations):
        fitness_values = [calculate_fitness(ind, distance_matrix) for ind in population]
        avg_distances.append(sum(fitness_values) / len(fitness_values))
        current_best = min(fitness_values)
        
        if current_best < best_fitness:
            best_fitness = current_best
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        
        if stagnant_generations >= generations // 10:
            break
        
        if generation % 10 == 0:
            print(f"Generation {generation}/{generations}: Avg Fitness = {avg_distances[-1]:.2f}")
        
        parents = [tournament_selection(population, fitness_values, tournament_size) for _ in range(population_size)]
        offspring = []
        
        for i in range(0, population_size, 2):
            if random.random() < crossover_rate:
                child1 = crossover(parents[i], parents[i+1])
                child2 = crossover(parents[i+1], parents[i])
            else:
                child1, child2 = parents[i], parents[i+1]
            offspring.append(mutation(child1, mutation_rate))
            offspring.append(mutation(child2, mutation_rate))
        
        population = offspring
    
    best_index = min(range(len(fitness_values)), key=lambda i: fitness_values[i])
    best_route = population[best_index]
    execution_time = time.time() - start_time
    print(f"Genetic Algorithm completed in {execution_time:.2f} seconds")
    
    return best_fitness, avg_distances, best_route, execution_time, generation

# ==========================
# Grid Search and Execution
# ==========================

def run_grid_search(dataset):
    """Runs grid search on a dataset using multiprocessing."""
    print(f"Starting grid search for {dataset}")
    city_coords = prepare_tsp_files(dataset)
    distance_matrix = compute_distance_matrix(city_coords)
    
    # first run berlin52
    # crossover_methods = [ordered_crossover, pmx_crossover]
    # mutation_methods = [swap_mutation, scramble_mutation]
    # population_sizes = [100, 200, 500]
    # num_generations = [100, 500, 1000, 2000]
    # mutation_rates = [0.01, 0.1]
    # crossover_rates = [0.1, 0.5, 1.0]
    
    # berlin to do
    # run best case again but don't limit generations; see if there's an improvement for report.

    # for kroa100 run 1 
    # crossover_methods = [ordered_crossover]
    # mutation_methods = [swap_mutation]
    # population_sizes = [500]
    # num_generations = [500, 1000, 2000]
    # mutation_rates = [0.01, 0.1]
    # crossover_rates = [0.1, 0.5, 1.0]\

    #  test run of pr1002 with optimal from other 2 datasets, in theory the 'optimal' solution
    # also reurunning for berlin52 with no stopping fn for generations
    # crossover_methods = [ordered_crossover]
    # mutation_methods = [swap_mutation]
    # population_sizes = [500]
    # num_generations = [2000]
    # mutation_rates = [0.1]
    # crossover_rates = [1.0]

    # trying to get better fitness for berlin52
    # crossover_methods = [ordered_crossover]
    # mutation_methods = [swap_mutation]
    # population_sizes = [5000]
    # num_generations = [1000]
    # mutation_rates = [0.05]
    # crossover_rates = [1.0]
    # tournament_size = [2, 3, 5, 8, 10]

    # let's do a final big grid search for berlin 52 to get our final optimal GA which we will test on all 3 datasets
    # this is 810 permutations apparently which is a lot.
    # crossover_methods = [ordered_crossover]
    # mutation_methods = [swap_mutation]
    # population_sizes = [100, 200, 500, 1000, 2000, 5000]
    # num_generations = [1000, 2000, 5000]
    # mutation_rates = [0.01, 0.05, 0.1]
    # crossover_rates = [0.1, 0.5, 1.0]
    # tournament_size = [2, 3, 5, 8, 10]

    # now we have our best grid. let's ignore everything else and just run this.
    # running 3 times to get a good average
    # crossover_methods = [ordered_crossover]
    # mutation_methods = [swap_mutation]
    # population_sizes = [2000]
    # num_generations = [5000]
    # mutation_rates = [0.01]
    # crossover_rates = [1.0]
    # tournament_size = [5]

    # first tried low pop high gen, now swapping for comparison. 
    crossover_methods = [ordered_crossover]
    mutation_methods = [swap_mutation]
    population_sizes = [2000]
    num_generations = [500]
    mutation_rates = [0.1]
    crossover_rates = [0.5]

    tournament_size = [5]


    configurations = list(itertools.product(crossover_methods, mutation_methods, population_sizes, num_generations, mutation_rates, crossover_rates, tournament_size))
    results = []
    best_overall = float('inf')
    best_result = None

    for cfg in tqdm.tqdm(configurations, desc=f"Processing {dataset}", unit="config"):
        crossover, mutation, pop_size, generations, mutation_rate, crossover_rate, tournament_size = cfg
        best_distance, avg_distances, best_route, execution_time, final_gen = genetic_algorithm(
            distance_matrix, pop_size, generations, mutation_rate, crossover_rate, crossover, mutation, tournament_size
        )
        
        results.append({
            "Dataset": dataset,
            "Crossover Rate": crossover_rate,
            "Crossover": crossover.__name__,
            "Mutation": mutation.__name__,
            "Population Size": pop_size,
            "Generations": final_gen,
            "Mutation Rate": mutation_rate,
            "Tournament Size": tournament_size,
            "Best Distance": round(best_distance, 2),
            "Execution Time (s)": round(execution_time, 2)
        })
        
        if best_distance < best_overall:
            best_overall = best_distance
            best_result = (final_gen, best_route, avg_distances, round(best_distance, 2))
    
    df_results = pd.DataFrame(results)
    save_results_to_file(df_results, dataset)
    save_best_result(dataset, *best_result)
    print(f"Grid search for {dataset} completed.")

if __name__ == "__main__":
    datasets = ["kroa100.txt", "pr1002.txt"]
    print("Starting multiprocessing grid search...")
    with multiprocessing.Pool(processes=len(datasets)) as pool:
        pool.map(run_grid_search, datasets)
    print("Grid search finished.")