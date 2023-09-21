import random
import itertools
import multiprocessing
import matplotlib.pyplot as plt
from random import randint


# class to create each individual in inputs population
class Create_Input:
    def __init__(self, permutation):
        self.permutation = permutation
        self.fitness = -1

# class to create population of permutation
class Inputs:
    def __init__(self, numbers, mutation_rate, elitism_percent, POPSIZE):
        self.numbers = numbers
        self.population = []
        self.new_population = []
        self.mutation_rate = mutation_rate
        self.ELITISM_SIZE = elitism_percent*POPSIZE
        self.POPSIZE = POPSIZE

    def init_population(self):
        for _ in range(self.POPSIZE):
            random_permutation = [i for i in range(1, self.numbers+1)]
            random.shuffle(random_permutation)
            individual = Create_Input(random_permutation)
            self.population.append(individual)

    def PMX(self, parent1, parent2):
        size = len(parent1)
        # Choosing two crossover points
        a, b = sorted(random.sample(range(size), 2))

        # Copying parents to offspring
        offspring1, offspring2 = parent1[:], parent2[:]

        # Swapping segments
        offspring1[a:b], offspring2[a:b] = offspring2[a:b], offspring1[a:b]

        # Resolving conflicts in the offspring
        for i in range(size):
            if i < a or i >= b:
                while offspring1[i] in offspring1[a:b]:
                    idx = parent2.index(offspring1[i])
                    offspring1[i] = parent1[idx]

                while offspring2[i] in offspring2[a:b]:
                    idx = parent1.index(offspring2[i])
                    offspring2[i] = parent2[idx]

        # Return the offspring
        return Create_Input(offspring1), Create_Input(offspring2)

    #pick 3 positions randomly and then inverse between two indexes and then change place according to third index
    def inversion_mutation(self, individual):
        if random.random() < self.mutation_rate:
            pos1, pos2 = sorted([random.randint(0, self.numbers-1), random.randint(0, self.numbers-1)])
            mov_pos = random.randint(pos2, self.numbers-1)
            individual = individual[0:pos1] + individual[pos2:mov_pos] + individual[pos1:pos2][::-1] + individual[mov_pos:]
        return individual


    def elitism(self):
        for i in range(int(self.ELITISM_SIZE)):
            self.new_population.append(self.population[i])

    def swap(self):
        self.population = self.new_population
        self.new_population = []

    def fitness(self, good_networks):
        # The more networks an input fails, the higher its fitness
        for individual in self.population:
            individual.fitness = sum(int(self.check_network_failure(network, individual.permutation)) for network in good_networks)

    def check_network_failure(self, network, input):
        # Use the sorting network to sort the input
        sorted_input = self.apply_network(network, input)

        # Return True if the sorting network failed to sort the input
        return sorted_input != sorted(input)

    def apply_network(self, network, input):
        # Apply a sorting network to an input
        for (i, j) in network:
            if input[i] > input[j]:
                input[i], input[j] = input[j], input[i]
        return input


    def get_best_solution(self):
        best_solution = min(self.population, key=lambda individual: individual.fitness)
        return best_solution

    def genetic_algorithm(self):
        self.elitism()
        while len(self.new_population) < self.POPSIZE:
            #choose parents
            parent1_index = randint(0, self.POPSIZE // 2)
            parent2_index = randint(0, self.POPSIZE // 2)
            parent1 = self.population[parent1_index].permutation
            parent2 = self.population[parent2_index].permutation
            #crossover
            child1, child2 = self.PMX(parent1, parent2)
            #mutation
            child1.permutation = self.inversion_mutation(child1.permutation)
            child2.permutation = self.inversion_mutation(child2.permutation)

            #add child to population
            self.new_population.append(child1)
            self.new_population.append(child2)

    def get_top_individuals(self, x):
        # Sort the population based on fitness
        self.population.sort(key=lambda individual: individual.fitness, reverse=True)
        # Return the top x individuals
        return [individual.permutation for individual in self.population[:x]]


class Network:
    def __init__(self, size, num_comparators, comparators=None):
        self.size = size
        self.num_comparators = num_comparators
        self.divided_comparators = []
        if comparators is None:
            self.generate_initial_comparators()
            self.apply_target_depth_heuristic()
        else:
            self.comparators = comparators


    def generate_initial_comparators(self):
        comparators = []

        if self.size == 6:
            #Bitonic network for 6 numbers (9 comparators)
            comparators = [
                (0, 1), (2, 3), (4, 5),
                (0, 2), (1, 3), (3, 4),
                (1, 2), (3, 5), (2, 4)
            ]
        if self.size == 7:
            comparators = [
                (0, 1), (2, 3), (4, 5),(0, 2), (1, 3), (4, 6),
                (1, 2), (3, 5), (4, 5),
                (0, 4), (1, 5), (2, 6),
                (2, 4), (3, 6), (1, 4), (3, 5)
            ]
        elif self.size == 16:
            # Bitonic network for 16 numbers (19 comparators)
            comparators = [
                (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
                (10, 11), (12, 13), (14, 15), (0, 2),
                (1, 3), (5, 7), (9, 11), (13, 15),
                (0, 8), (1, 9), (4, 12), (5, 13),
                (6, 14), (7, 15)
            ]



        # Generate remaining random comparators
        remaining_comparators = self.num_comparators - len(comparators)
        for _ in range(remaining_comparators):
            a, b = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            # Ensure i != j and (i, j) and (j, i) are not already in comparators
            while (a == b) or ((a, b) in comparators) or ((b, a) in comparators):
                a, b = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            comparators.append((a, b))
        self.comparators = comparators

    def apply_target_depth_heuristic(self):
        depth = self.calculate_depth()
        max_diff = -1
        max_pair = None
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                if abs(depth[i] - depth[j]) > max_diff:
                    max_diff = abs(depth[i] - depth[j])
                    max_pair = (i, j) if depth[i] < depth[j] else (j, i)
        self.comparators.append(max_pair)

    def calculate_depth(self):
        depth = [0] * self.size
        for a, b in self.comparators:
            depth[b] = max(depth[b], depth[a] + 1)
        return depth

    def sort(self, data):
        data = list(data)
        for a, b in self.comparators:
            if data[a] > data[b]:
                data[a], data[b] = data[b], data[a]
        return data

    def add_comparator(self):
        a, b = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        if a != b and (a, b) not in self.comparators and (b, a) not in self.comparators:
            self.comparators.append((a, b))

    def remove_comparator(self):
        if len(self.comparators) > self.num_comparators:
            self.comparators.pop(random.randint(0, len(self.comparators) - 1))

    def swap_comparators(self):
        if len(self.comparators) > 1:
            i, j = random.sample(range(len(self.comparators)), 2)
            self.comparators[i], self.comparators[j] = self.comparators[j], self.comparators[i]

    def reverse_comparators(self):
        if len(self.comparators) > 1:
            i, j = sorted(random.sample(range(len(self.comparators)), 2))
            self.comparators[i:j+1] = reversed(self.comparators[i:j+1])

    def divide_comparators_into_groups(self):
        grouped_comparators = []
        seen_indices = set()

        for comparator in self.comparators:
            i, j = comparator

            # Check if i or j has been seen before (dependency)
            if i in seen_indices or j in seen_indices:
                # Create a new group
                grouped_comparators.append([comparator])
                seen_indices.clear()
                seen_indices.update(comparator)
            else:
                # Add the comparator to the current group
                if len(grouped_comparators) == 0:
                    grouped_comparators.append([])
                grouped_comparators[-1].append(comparator)
                seen_indices.update(comparator)

        self.divided_comparators = grouped_comparators


class Population:
    def __init__(self, population_size, network_size , num_comparators):
        self.num_comparators = num_comparators
        self.networks = [Network(network_size,self.num_comparators) for _ in range(population_size)]
        self.prev_avg_fitness = 0
        self.generation_count = 0

    def fitness(self, network, test_data):
        return sum(int(network.sort(data) == sorted(data)) for data in test_data)


    def selection(self, test_data, tournament_size=3):
        parents = []
        for _ in range(len(self.networks)):
            tournament = random.sample(self.networks, tournament_size)
            winner = max(tournament, key=lambda network: self.fitness(network, test_data))
            parents.append(winner)
        return random.choice(parents)

    def crossover(self, parent1, parent2):
        split = random.randint(0, parent1.size - 1)
        child_comparators = parent1.comparators[:split] + parent2.comparators[split:]
        return Network(parent1.size,self.num_comparators, child_comparators)

    def mutate(self, network, rate):
        mutated_network = Network(network.size,self.num_comparators, network.comparators[:])
        for _ in range(len(network.comparators)):
            if random.random() < rate:
                choice = random.choice(['add', 'remove', 'swap', 'reverse'])
                if choice == 'add':
                    mutated_network.add_comparator()
                elif choice == 'remove':
                    mutated_network.remove_comparator()
                elif choice == 'swap':
                    mutated_network.swap_comparators()
                elif choice == 'reverse':
                    mutated_network.reverse_comparators()
        return mutated_network

    def evolve(self, test_data, mutation_rate, elitism_rate=0.2, global_mutation_rate=0.3, global_mutation_trigger=10):
        new_networks = []
        avg_fitness = 0  # Initialize an average fitness value

        # Apply elitism: select a percentage of the best networks and keep them in the new generation
        num_elites = int(len(self.networks) * elitism_rate)
        elites = sorted(self.networks, key=lambda network: self.fitness(network, test_data), reverse=True)[:num_elites]
        new_networks.extend(elites)

        if self.generation_count % 10 == 0:  # Adjust mutation rate every 10 generations
            if self.generation_count <= 50:
                mutation_rate = 0.7  # Higher mutation rate in the initial generations
            else:
                mutation_rate = 0.3  # Lower mutation rate in later generations

        # Generate offspring using selection, crossover, and mutation
        num_offspring = len(self.networks) - num_elites
        offspring_args = [(self.selection, self.crossover, self.mutate, mutation_rate, test_data) for _ in
                          range(num_offspring)]

        with multiprocessing.Pool() as pool:
            offspring = pool.starmap(self.create_offspring, offspring_args)

        new_networks.extend(offspring)
        avg_fitness += sum(self.fitness(child, test_data) for child in offspring) / len(self.networks)

        # Perform a global mutation if the fitness has not significantly improved
        if abs(self.prev_avg_fitness - avg_fitness) < global_mutation_trigger:
            for network in new_networks:
                if random.random() < global_mutation_rate:
                    self.mutate(network, mutation_rate)

        self.prev_avg_fitness = avg_fitness
        self.networks = new_networks

    def create_offspring(self, selection_func, crossover_func, mutation_func, mutation_rate, test_data):
        parent1 = selection_func(test_data)
        parent2 = selection_func(test_data)
        child = crossover_func(parent1, parent2)
        child = mutation_func(child, mutation_rate)
        return child

    def get_top_networks(self, x, test_data):
        # Sort the networks based on fitness
        self.networks.sort(key=lambda network: self.fitness(network, test_data), reverse=True)
        # Return the top x networks
        return self.networks[:x]


def generate_test_data(network_size):
    # Generate all sequences of 0's and 1's of length network_size
    return [list(seq) for seq in itertools.product([0, 1], repeat=network_size)]

def create_histogram(generations, best_fitness_values, num_comparators_values):
    # Plot the best fitness for each generation
    plt.plot(generations, best_fitness_values, 'o-', label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    # Plot the number of comparators for each generation
    plt.plot(generations, num_comparators_values, 'o-', color='orange', label='Number of Comparators')
    plt.ylabel('Number of Comparators and Fitness values')

    # Set the title and legends
    plt.title('Fitness and Number of Comparators per Generation')
    plt.legend()

    # Show the plot
    plt.show()

def create_histogram2(generations, best_fitness_values):
    # Plot the best fitness for each generation
    plt.plot(generations, best_fitness_values, '-o', label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    plt.title('Fitness per Generation')
    plt.legend()

    plt.show()



def main():
    network_size = int(input("choose the size you want 6 or 16 : "))
    if network_size == 6:
        population_size = 100
        mutation_rate = 0.5
        num_generations = 100
        samples = 64
        num_comparators = 12
        input_evolve_rate = 1
        network_evolve_rate = 8

    elif network_size == 7:
        population_size = 200
        mutation_rate = 0.5
        num_generations = 100
        samples = 128
        num_comparators = 16
        input_evolve_rate = 1
        network_evolve_rate = 5

    else:
        population_size = 200
        mutation_rate = 0.5
        num_generations = 100
        samples = 100
        num_comparators = 61
        input_evolve_rate = 10
        network_evolve_rate = 20

    samples2 = int(population_size / 2)


    population = Population(population_size, network_size, num_comparators)

    best_global_fitness = -1  # Initial value
    best_global_network = None  # Initial value
    input_sequence = Inputs(network_size,0.5,0.2,population_size)

    input_sequence.init_population()
    test_data = [individual.permutation for individual in random.sample(input_sequence.population, samples)]

    best_fitness_values = []
    num_comparators_values = []
    best_fitness_input = []
    for generation in range(num_generations):
        print(f'Generation {generation+1}')
        population.generation_count = generation+1
        if generation % network_evolve_rate == 0:
            population.evolve(test_data, mutation_rate)
        best_network = max(population.networks, key=lambda network: population.fitness(network, test_data))
        best_fitness = population.fitness(best_network, test_data)
        print(f'Best fitness: {best_fitness}, Number of comparators: {len(best_network.comparators)}')
        best_fitness_values.append(best_fitness)
        num_comparators_values.append(len(best_network.comparators))
        current_generation = generation
        # Update the best global fitness and network if current best is better
        if best_fitness >= best_global_fitness:
            best_global_fitness = best_fitness
            best_global_network = best_network
        if best_fitness == 64 and len(best_network.comparators) == 12: break

        best_networks = [network.comparators for network in population.get_top_networks(samples2, test_data)]
        if generation % input_evolve_rate == 0:
            input_sequence.genetic_algorithm()
            input_sequence.fitness(best_networks)
            input_sequence.swap()

        best_inputs = input_sequence.get_top_individuals(samples)
        test_data = best_inputs
        input_sequence.population.sort(key=lambda individual: individual.fitness, reverse=True)
        best_fitness_input.append(input_sequence.population[0].fitness)

        print()
        # Print the best global fitness and its number of comparators at the end
    print()
    print(f'Best global fitness: {best_global_fitness}, Number of comparators in best global network: {len(best_global_network.comparators)}')
    best_global_network.divide_comparators_into_groups()
    print(f'Best comparators: {best_global_network.divided_comparators}')
    generations = range(1, current_generation+2)
    create_histogram(generations, best_fitness_values, num_comparators_values)
    generations = range(1, current_generation + 1)
    create_histogram2(generations,best_fitness_input)


if __name__ == '__main__':
    main()