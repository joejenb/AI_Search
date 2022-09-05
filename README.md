## Overview 

Had to implement two different baseline algorithms to solve the travelling salesman problem. In addition to this, also had to implement two further algorithms that built on these baselines.

## Genetic Algorithm Enhancements

**Initial population generation:** I use the Nearest Neighbour algorithm once for each possible
starting city in order to generate the initial population. If this gives less than 200 members then
the algorithm repeatedly mutates already present members until the population is large
enough.

**Elite survival:** The best 100 tours from the parent population will carry to the child population,
allowing mutation while maintaining a population with high fitness.

**Edge Recombination(ER) operator:** Tries to preserve the edges common to both parents,
treating them as the reason for why they both have good fitness.

**Simple Inversion Mutation(SIM):** Reverses the substring between two randomly selected cut
points, therefore, only changing two edges within the tour.

## Hill Climbing Enhancements

To increase the rate of climbing, instead of computing the tour of every node in the
neighbourhood of the current node, as soon as a node with shorter tour length is found it
becomes the current node. The computation of the neighbourhood then continues but now for
the new current node. This is the most significant improvement.

I also changed the algorithm so that multiple iterations can occur, each with a new randomly
generated tour.

Thinking that tours already encountered will likely pull the algorithm towards the same local
maxima, I made a set to hold all tours that had been computed. However, for larger city sets this
required too much memory, therefore, the algorithm only stores every other tour computed.
Tours generated at the start of each iteration are then shuffled until an unencountered tour is
found.
