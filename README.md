# SortingNetwork_AISolution
![python](https://img.shields.io/badge/Language-Python-pink)

**Coevolution approach to the sorting network problem.**


**Background:**

Coevolution is an evolutionary computation technique where multiple populations evolve together, typically in a competitive environment. The evolution of one population is dependent on the evolution of the other, and they influence each other's fitness landscapes. This approach can be particularly effective in problems where the solution space is dynamic and dependent on multiple factors or agents.

In the context of the sorting network problem, coevolution involves two populations: sorting networks and inputs. The goal is to evolve optimal sorting networks that can correctly sort any given inputs, while simultaneously evolving inputs that challenge the sorting networks to ensure their robustness.

This project explores the use of coevolutionary algorithm to identify optimal sorting networks based on the foundational work of Hili's experiment. The goal is to determine the best sorting network configurations and the number of comparators required for different input sizes.

**Approach:**

1. Evolution of Sorting Networks:
1.1.Selection: Tournament selection was employed, where a subset of individuals is chosen, and the best among them, based on fitness, is selected for the next generation.

1.2.Crossover: A single-point crossover method was used. This involves selecting a random crossover point and creating a child by taking the genes from the first parent up to the crossover point, and then from the second parent after the crossover point.

1.3.Mutation: Multiple mutation methods were implemented to introduce genetic diversity:

1.3.1.Add Comparator: Introduce a new comparator to the network.
1.3.2.Remove Comparator: Remove an existing comparator from the network.
1.3.3.Swap Comparators: Swap the positions of two comparators.
1.3.4.Reverse Comparators: Reverse a sequence of comparators in the network.

Through these evolutionary techniques, the aim was to produce an optimal sorting network capable of handling a wide range of inputs.

2. Evolution of Inputs:
Inputs were treated as permutations, and the goal was to evolve inputs that posed challenges for the sorting networks, ensuring that the evolved networks were robust and versatile.Similar to the sorting networks, a selection,crossover,mutation mechanisms were employed.

By evolving challenging inputs, the sorting networks were constantly tested, ensuring that only the most robust networks progressed through the generations.


**Highlights:**

1.Built upon Hili's experiment results to set benchmarks for optimal sorting network configurations.

2.Successfully identified optimal sorting networks for inputs of 6, 7, and 8 numbers.

3.Achieved a sorting network configuration for an input size of 16 numbers that correctly sorted 70% of the test cases.

Run the program:

For input of 6/7 numbers run 6_numbers_solution.py by choosing 6 or 7 as input .

For input of 16 numbers run 16_numbers_solution.py by choosing 16 .

Note : The separation is made because running input of 16 numbers have much higher complexity then running input of 6 numbers.
