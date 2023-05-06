import math
import random

class Solution:
    def __init__(self, randomize_genes = False):
        self.genes = [0] * 32
        if randomize_genes:
            for i in range(32):
                if random.random() < 0.5:
                    self.genes[i] = 0
                else:
                    self.genes[i] = 1
    def get_adaptation(self):
        self.fix_values()
        x = int("".join(str(bit) for bit in self.genes[0:8]), 2)
        y = int("".join(str(bit) for bit in self.genes[8:16]), 2)
        z = int("".join(str(bit) for bit in self.genes[16:24]), 2)
        w = int("".join(str(bit) for bit in self.genes[24:]), 2)

        return 5000 - (600 * (x-20)**2 * (y-35)**2) / z - (x-50)**2 * (w-48)**2 + w

    def crossover(self, other_solution):
        cut_position = random.randint(0,15)
        new_solution = Solution()
        new_solution.genes[0:cut_position] = self.genes[0:cut_position]
        new_solution.genes[cut_position:] = other_solution.genes[cut_position:]
        return new_solution
    
    def mutation(self):
        mutation_position = random.randint(0,15)
        if self.genes[mutation_position] == 1:
            self.genes[mutation_position] = 0
        else:
            self.genes[mutation_position] = 1
    
    def fix_values(self):
        x = int("".join(str(bit) for bit in self.genes[0:8]), 2)
        y = int("".join(str(bit) for bit in self.genes[8:16]), 2)
        z = int("".join(str(bit) for bit in self.genes[16:24]), 2)
        w = int("".join(str(bit) for bit in self.genes[24:]), 2)
        if x==0:
            mutation_position = random.randint(0,7)
            self.genes[mutation_position] = 1
        if y==0:
            mutation_position = random.randint(8,15)
            self.genes[mutation_position] = 1
        if z==0:
            mutation_position = random.randint(16,23)
            self.genes[mutation_position] = 1
        if w==0:
            mutation_position = random.randint(24,31)
            self.genes[mutation_position] = 1
    
def GetWeights(adaptations, population_size, best_population_size):
    adaptations_copy = adaptations.copy()
    minValue = min(adaptations)
    weights = [0] * population_size
    reproduction_chance = 1.0 / best_population_size
    for i in range(0,best_population_size):
        index = adaptations_copy.index(max(adaptations_copy))
        adaptations_copy[index] = minValue - 1
        weights[index] = reproduction_chance
    return weights

def Solve():
    population_size = 180
    iterations = 100
    mutation_chance = 0.5
    threshold_value = 0.5
    population = [Solution(randomize_genes = True) for i in range(population_size)]
    best_solution = Solution()
    best_solution_adaptation = 0.
    best_iteration_found = 0
    for iteration in range(iterations):
        adaptations = [p.get_adaptation() for p in population]
        local_best_solution = population[adaptations.index(max(adaptations))]
        if local_best_solution.get_adaptation() > best_solution_adaptation:
            best_solution = local_best_solution
            best_solution_adaptation = local_best_solution.get_adaptation()
            best_iteration_found = iteration
        weights = GetWeights(adaptations, population_size, math.ceil(threshold_value * population_size))
        parents = [random.choices(population, weights=weights, k = 2) for i in range(population_size)]
        children = [p[0].crossover(p[1]) for p in parents]
        for c in children:
            if random.random() < mutation_chance:
                c.mutation()
            population = children
    adaptations = [p.get_adaptation() for p in population]
    local_best_solution = population[adaptations.index(max(adaptations))]
    if local_best_solution.get_adaptation() > best_solution_adaptation:
        best_solution = local_best_solution
        best_solution_adaptation = local_best_solution.get_adaptation()
    x = int("".join(str(bit) for bit in best_solution.genes[0:8]), 2)
    y = int("".join(str(bit) for bit in best_solution.genes[8:16]), 2)
    z = int("".join(str(bit) for bit in best_solution.genes[16:24]), 2)
    w = int("".join(str(bit) for bit in best_solution.genes[24:]), 2)
    return x, y, z, w, best_solution_adaptation

x, y, z, w, value = Solve()
print('x =', x, ', y =', y, ', z =', z,', w =', w,', largest value =', value)