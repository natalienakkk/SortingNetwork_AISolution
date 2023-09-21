import random
import itertools
import matplotlib.pyplot as plt
import multiprocessing

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
        elif self.size == 7:
            comparators = [
                (0, 1), (2, 3), (4, 5),(0, 2), (1, 3), (4, 6),
                (1, 2), (3, 5), (4, 5),
                (0, 4), (1, 5), (2, 6),
                (2, 4), (3, 6), (1, 4), (3, 5)
            ]
        elif self.size == 16:
        # Bitonic network for 16 numbers (19 comparators)
            comparators = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
                (10, 11), (12, 13), (14, 15), (0, 2),
                (1, 3), (5, 7), (9, 11), (13, 15),
                (0, 8), (1, 9), (4, 12), (5, 13),
                (6, 14), (7, 15) ]

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


# def generate_test_data(network_size):
#     # Generate all sequences of 0's and 1's of length network_size
#     if network_size == 16:
#         return random.sample([list(seq) for seq in itertools.product([0, 1], repeat=network_size)], 2000)
#     else:
#         return [list(seq) for seq in itertools.product([0, 1], repeat=network_size)]

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

def main():
    network_size = int(input("choose the size you want 6 or 16 : "))
    if network_size == 16:
        network_size = 16
        population_size = 500
        mutation_rate = 0.25
        num_generations = 120
        num_comparators = 64

    elif network_size == 7:
        population_size = 200
        mutation_rate = 0.5
        num_generations = 100
        samples = 128
        num_comparators = 16
        input_evolve_rate = 1
        network_evolve_rate = 5

    else:
        network_size = 6
        population_size = 100
        mutation_rate = 0.25
        num_generations = 100
        num_comparators = 12



    # Generate all permutations of 0's and 1's of length network_size
    # test_data = prem.prems()
    test_data = generate_test_data(network_size)


    population = Population(population_size, network_size , num_comparators)

    best_global_fitness = -1  # Initial value
    best_global_network = None  # Initial value
    best_fitness_values = []
    num_comparators_values = []

    for generation in range(num_generations):
        print(f'Generation {generation+1}')
        population.evolve(test_data, mutation_rate)
        best_network = max(population.networks, key=lambda network: population.fitness(network, test_data))
        best_fitness = population.fitness(best_network, test_data)
        print(f'Best fitness: {best_fitness}, Number of comparators: {len(best_network.comparators)}')
        best_fitness_values.append(best_fitness)
        num_comparators_values.append(len(best_network.comparators))
        current_generation = generation
        # Update the best global fitness and network if the current best is better
        if best_fitness >= best_global_fitness:
            best_global_fitness = best_fitness
            best_global_network = best_network
        if best_fitness == 64 and len(best_network.comparators) == 12 :break
        print()


    # Print the best global fitness and the number of comparators in the best global network at the end
    print(f'Best global fitness: {best_global_fitness}, Number of comparators in the best global network: {len(best_global_network.comparators)}')
    best_global_network.divide_comparators_into_groups()
    print(f'Best comparators: {best_global_network.divided_comparators}')
    generations = range(1, current_generation+2)
    create_histogram(generations, best_fitness_values, num_comparators_values)


if __name__ == '__main__':
    main()


