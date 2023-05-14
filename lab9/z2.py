import random
import math
import numpy as np
import matplotlib.pyplot as plt

population_size = 100
chromosome_length = 20
num_generations = 100

def function(x, y):
    return x ** 2 + y ** 2 - 20 * (math.cos(math.pi * x) + math.cos(math.pi * y) - 2)

def generate_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        chromosome = ''.join(random.choice('01') for _ in range(chromosome_length))
        population.append(chromosome)
    return population

def bin_to_gray(binary):
    gray = binary[0]
    for i in range(1, len(binary)):
        gray += str(int(binary[i - 1]) ^ int(binary[i]))
    return gray

def gray_to_bin(gray):
    binary = gray[0]
    for i in range(1, len(gray)):
        binary += str(int(binary[i - 1]) ^ int(gray[i]))
    return binary

def decode(chromosome, encoding='binary'):
    if encoding == 'gray':
        chromosome = gray_to_bin(chromosome)
    return int(chromosome, 2)

def fitness(chromosome):
    x = decode(chromosome) / (2**chromosome_length - 1)
    return function(x)

def roulette_wheel_selection(population, num_selected):
    scores = [fitness(chromosome) for chromosome in population]
    total_score = sum(scores)
    probabilities = [score / total_score for score in scores]

    selected_chromosomes = np.random.choice(population, size=num_selected, p=probabilities, replace=False)
    return list(selected_chromosomes)

def threshold_selection(population, num_selected):
    scores = [(chromosome, fitness(chromosome)) for chromosome in population]
    threshold = np.percentile([score for _, score in scores], 100 - (100 * num_selected / len(population)))
    
    return [chromosome for chromosome, score in scores if score >= threshold]

def crossover(population, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.choices(population, k=2)
        crossover_point = random.randint(0, chromosome_length)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        new_population.extend([child1, child2])
    return new_population

def mutation(population, mutation_probability):
    for i in range(len(population)):
        chromosome = list(population[i])
        for j in range(len(chromosome)):
            if random.random() < mutation_probability:
                chromosome[j] = '0' if chromosome[j] == '1' else '1'
        population[i] = ''.join(chromosome)
    return population

def evaluate_algorithm(selection_method, encoding):
    population = generate_population(population_size, chromosome_length)
    best_fitness_values = []

    for _ in range(num_generations):
        if selection_method == 'roulette_wheel':
            best_individuals = roulette_wheel_selection(population, num_best)
        elif selection_method == 'threshold':
            best_individuals = threshold_selection(population, num_best)
        else:
            raise ValueError("Unknown selection method")

        new_population = crossover(best_individuals, population_size)
        population = mutation(new_population, mutation_probability)

        best_chromosome = threshold_selection(population, 1)[0]
        best_fitness = fitness(best_chromosome)
        best_fitness_values.append(best_fitness)

    best_x = decode(best_chromosome, encoding) / (2**chromosome_length - 1)
    best_value = fitness(best_chromosome)

    return best_x, best_fitness_values

def plot_progress(best_fitness_values, encoding):
    plt.plot(best_fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title(f"Genetic Algorithm Progress with {encoding} Encoding")
    plt.show()
  
def evaluate_algorithm_no_plot(selection_method, encoding):
    population = generate_population(population_size, chromosome_length)

    for _ in range(num_generations):
        if selection_method == 'roulette_wheel':
            best_individuals = roulette_wheel_selection(population, num_best)
        elif selection_method == 'threshold':
            best_individuals = threshold_selection(population, num_best)
        else:
            raise ValueError("Unknown selection method")

        new_population = crossover(best_individuals, population_size)
        population = mutation(new_population, mutation_probability)

    best_chromosome = threshold_selection(population, 1)[0]
    best_x = decode(best_chromosome, encoding) / (2**chromosome_length - 1)
    best_value = fitness(best_chromosome)

    return best_x, best_value

mutation_probability = 1
num_best = 50
print(f"Mutation probability: {mutation_probability}, Gamma: {num_best}")
best_x_binary = best_value_binary = best_x_gray = best_value_gray = best_x_roulette = best_value_roulette = 0

for i in range(10):
    temp_best_x_binary, temp_best_value_binary = evaluate_algorithm_no_plot('threshold', 'binary')
    best_x_binary += temp_best_x_binary
    best_value_binary += temp_best_value_binary
    temp_best_x_gray, temp_best_value_gray = evaluate_algorithm_no_plot('threshold', 'gray')
    best_x_gray += temp_best_x_gray
    best_value_gray += temp_best_value_gray
    temp_best_x_roulette, temp_best_value_roulette = evaluate_algorithm_no_plot('roulette_wheel', 'binary')
    best_x_roulette += temp_best_x_roulette
    best_value_roulette += temp_best_value_roulette

best_x_binary /= 10
best_value_binary /= 10
best_x_gray /= 10
best_value_gray /= 10
best_x_roulette /= 10
best_value_roulette /= 10

print(f"Binary encoding, threshold {num_best}% - Best x: {best_x_binary}, Best f(x) value: {best_value_binary}")
print(f"Gray encoding, threshold {num_best}% - Best x: {best_x_gray}, Best f(x) value: {best_value_gray}")
print(f"Roulette selection - Best x: {best_x_roulette}, Best f(x) value: {best_value_roulette}")