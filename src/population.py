import creature 
import numpy as np
import genome
import random

class Population:
    def __init__(self, pop_size, gene_count):
        self.creatures = [creature.Creature(
                          gene_count=gene_count) 
                          for i in range(pop_size)]

    @staticmethod
    def get_fitness_map(fits):
        fitmap = []
        total = 0
        for f in fits:
            total = total + f
            fitmap.append(total)
        return fitmap
    
    @staticmethod
    def select_parent(fitmap):
        r = np.random.rand() # 0-1
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i
    
    def evolve(self, elite_fraction=0.4, mutation_rate=0.1):
        self.creatures.sort(key=lambda c: c.fitness, reverse=True)

        # Ensure at least 2 elites
        elite_count = max(2, int(len(self.creatures) * elite_fraction))
        elites = self.creatures[:elite_count]

        new_creatures = elites.copy()
        while len(new_creatures) < len(self.creatures):
            parent1, parent2 = random.sample(elites, 2)
            child_genome = genome.crossover(parent1.genes, parent2.genes)
            genome.mutate(child_genome, mutation_rate)
            child = creature.Creature(genes=child_genome)
            new_creatures.append(child)

        self.creatures = new_creatures
        print(f"Child genome: {child_genome}")



