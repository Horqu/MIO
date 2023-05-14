import numpy as np
import matplotlib.pyplot as plt
import time

start = time.process_time()

# Define the objective function
def f(x, y):
    return x**2 + y**2 - 20*(np.cos(np.pi*x) + np.cos(np.pi*y) - 2)

# Define the particle class
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.best_personal_position = np.copy(self.position)
        self.best_personal_fitness = f(*self.position)

# Define the function to update the velocity
def update_velocity(particle, best_global_position, w=0.5, c1=2, c2=2):
    r1 = np.random.uniform(0, 1, 2)
    r2 = np.random.uniform(0, 1, 2)
    inertia = w * particle.velocity
    cognitive = c1 * r1 * (particle.best_personal_position - particle.position)
    social = c2 * r2 * (best_global_position - particle.position)
    return inertia + cognitive + social

# Define the function to update the position
def update_position(particle):
    return particle.position + particle.velocity

# Define the function to implement particle swarm optimization
def particle_swarm_optimization(bounds, n_particles, max_iterations):
    particles = [Particle(bounds) for _ in range(n_particles)]
    best_global_position = min([particle.position for particle in particles], key=lambda x: f(*x))
    history = []
    for _ in range(max_iterations):
        for particle in particles:
            particle.velocity = update_velocity(particle, best_global_position)
            particle.position = update_position(particle)
            particle.position = np.clip(particle.position, *bounds)  # Ensure the particles are within the bounds
            fitness = f(*particle.position)
            if fitness < particle.best_personal_fitness:
                particle.best_personal_position = np.copy(particle.position)
                particle.best_personal_fitness = fitness
        best_global_position = min([particle.best_personal_position for particle in particles], key=lambda x: f(*x))
        history.append(f(*best_global_position))
    return history

# Run the particle swarm optimization algorithm 10 times
bounds = [-10, 10]
n_particles = 100
max_iterations = 100
history_values = [particle_swarm_optimization(bounds, n_particles, max_iterations) for _ in range(10)]

# Calculate and print the average fitness value and the standard deviation
average_fitness = np.mean([history[-1] for history in history_values])
standard_deviation = np.std([history[-1] for history in history_values])
print(f"c1 = 2, c2 = 2, w = 0.5")
print(f"The average fitness value is: {average_fitness}")
print(f"The standard deviation of the fitness values is: {standard_deviation}")

print("PSO time: ", time.process_time() - start, "sec")

# Plot the history for each run
plt.plot(history_values[0])
plt.title('Convergence over iterations')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()
plt.show()
