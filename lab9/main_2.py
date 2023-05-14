import numpy as np
import matplotlib.pyplot as plt
import time

start = time.process_time()

# Define the objective function
def f(x, y):
    return x**2 + y**2 - 20*(np.cos(np.pi*x) + np.cos(np.pi*y) - 2)

# Define the function to generate an individual
def generate_individual(bounds):
    return np.random.uniform(bounds[0], bounds[1], 2)

# Define the function to generate a population
def generate_population(bounds, size):
    return np.array([generate_individual(bounds) for _ in range(size)])

# Define the function to calculate the fitness of an individual
def calculate_fitness(individual):
    return f(*individual)

# Define the function to perform tournament selection
def tournament_selection(population, tournament_size):
    tournament_indices = np.random.choice(len(population), tournament_size)
    tournament = population[tournament_indices]
    return min(tournament, key=calculate_fitness)

# Define the function to perform crossover
def crossover(parent1, parent2):
    return (parent1 + parent2) / 2

# Define the function to perform mutation
def mutate(individual, mutation_rate, bounds):
    if np.random.rand() < mutation_rate:
        return generate_individual(bounds)
    else:
        return individual

# Define the function to implement the genetic algorithm
def genetic_algorithm(bounds, population_size, max_generations, tournament_size, mutation_rate):
    population = generate_population(bounds, population_size)
    history = []
    for _ in range(max_generations):
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, bounds)
            new_population.append(child)
        population = np.array(new_population)
        best_individual = min(population, key=calculate_fitness)
        history.append(calculate_fitness(best_individual))
    return best_individual, history

# Run the genetic algorithm 10 times
bounds = [-10, 10]
population_size = 100
max_generations = 100
tournament_size = 5
mutation_rate = 0.01
fitness_values = []
history_values = []

for _ in range(10):
    best_individual, history = genetic_algorithm(bounds, population_size, max_generations, tournament_size, mutation_rate)
    fitness_values.append(calculate_fitness(best_individual))
    history_values.append(history)

# Calculate and print the average fitness value and the standard deviation
average_fitness = np.mean(fitness_values)
standard_deviation = np.std(fitness_values)
print(f"The average fitness value is: {average_fitness}")
print(f"The standard deviation of the fitness values is: {standard_deviation}")

print("Genetic time: ", time.process_time() - start, "sec")

# Plot the history for each run
for i, history in enumerate(history_values):
    plt.plot(history, label=f'Run {i+1}')
plt.title('Convergence over generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend
