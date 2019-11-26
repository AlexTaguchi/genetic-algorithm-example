# ~~~ Maximize the Sum of a List of Binary Integers ~~~ #
# Modules
from deap import algorithms, base, creator, tools
import numpy as np
import random

# Create a maximizing (1.0,) or minimizing (-1.0,) fitness that inherits the base fitness class
creator.create('FitnessMax', base.Fitness, weights=(1.0,))

# Create an fitness maximizing individual with attributes in the form of a list
creator.create('Individual', list, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()

# Register a boolean attribute that uniformly samples the range 0 or 1
toolbox.register('attr_bool', random.randint, 0, 1)

# Register an individual characterized by 100 attributes
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)

# Register a population that consists of a list of individuals
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Define the sum function to be maximized where the return must be an iterable
def binary_sum(individual):
    return sum(individual),

# Register the fitness function
toolbox.register('evaluate', binary_sum)

# Register the crossover operator
toolbox.register('mate', tools.cxTwoPoint)

# Mutation operator flips each attribute with a probability of 0.05
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)

# Select the fittest individual to breed out of three randomly drawn from the current generation
toolbox.register('select', tools.selTournament, tournsize=3)

# Create an initial population of 1000 individuals
population = toolbox.population(n=1000)

# Keep a record of the top 1000 unique individuals
hof = tools.HallOfFame(1000)

# Display fitness statistics for each generation
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('std', np.std)
stats.register('min', np.min)
stats.register('max', np.max)

# Set crossing probability, mutation probability, and number of generations
cxpb, mutpb, ngen = 0.5, 0.2, 50

# Perform genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof)

# Report final statistics
fitnesses = [ind.fitness.values[0] for ind in hof]
print('\nSummary:')
print(f'- Best Individual Fitness = {max(fitnesses):.2f}')
print(f'- Fitness of Top 100 Individuals = {np.mean(fitnesses):.2f} +/- {np.std(fitnesses):.2f}')
