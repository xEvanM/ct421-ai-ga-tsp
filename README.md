# Traveling Salesman Problem (TSP) Genetic Algorithm

## Overview
This repository is a University of Galway, CT421 Artifical Intelligence Project, completed by Evan Murphy. This project implements a Genetic Algorithm to solve the Traveling Salesman Problem (TSP) using various selection, crossover, and mutation methods. It supports TSPLIB format files and includes basic functionality for analysing the results.

## Setup Instructions
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```sh
pip install numpy pandas matplotlib tqdm scipy
```

### Running the Genetic Algorithm
1. Prepare your dataset in the TSPLIB format.
2. Place the dataset files (e.g., `berlin52.txt`, `kroa100.txt`) in the project directory.
3. Configure your paramater grid for grid search. near the bottom of the file.
4. Run the genetic algorithm with multiprocessing:
   ```sh
   python tsp.py
   ```
   The script will perform a grid search over different hyperparameter configurations and save the results to CSV files.

## Modifying Params
Parameters for the genetic algorithm can be modified in `tsp.py` inside the `run_grid_search` function:
- **Crossover Methods:** Change `crossover_methods` to use `ordered_crossover` or `pmx_crossover`.
- **Mutation Methods:** Change `mutation_methods` to use `swap_mutation` or `scramble_mutation`.
- **Population Size:** Adjust `population_sizes` to control the number of individuals per generation.
- **Generations:** Adjust `num_generations` to increase or decrease iterations.
- **Mutation Rate:** Adjust `mutation_rates` to control mutation probability.
- **Crossover Rate:** Adjust `crossover_rates` to control probability of crossover.
- **Tournament Size:** Adjust `tournament_size` to control selection pressure.

## Analysing Results
If you wish, after running the genetic algorithm, follow the example fns shown in my `tspanalysis.py` file and then run with:
```sh
python tsp_analysis.py
```
This script will:
- Load CSV results.
- Print stats (best, worst, and average distances).
- Identify the best and worst parameter configurations.
- Create plots showing:
  - avg distance over mutation rate
  - avg distance over crossover rate
  - avg distance over population size
  - distance change over generations for individual instances

## Output Files
- `*_results_*.csv`: Stores results for different hyperparameter settings.
- `best_*_*.txt`: Contains details of the best solution found.
- Plots/graphs generated during analysis.

## Final Note
- This project was time-consuming due to the computational complexity with larger data sizes. As such, the functions and scripts aren't as refined as I would like them to be.
- The analysis function requires manually writing some functional calls, and was written quickly to aid me in writing my final report.
- I've retained previous paramater grids, commented out in the tsp.py file. These were left in so that my thought process and trial and error could be captured throughout the process. 
