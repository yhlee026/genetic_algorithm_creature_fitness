import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import creature
import math
import os
import genome
import population

#  Global parameters 
ARENA_SIZE = 20          # Width and length of the square arena
MOUNTAIN_HEIGHT = 5       # Peak height of the Gaussian mountain
SPAWN_HEIGHT = 1.0        # Distance above the plane to spawn creatures
SPAWN_MARGIN = 1.0        # Horizontal margin from arena edge to spawn creatures

# Connect to physics client and set up environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load a static ground plane for stable collisions
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -10)

# Environment setup
def make_mountain(num_rocks=200, max_size=0.5, arena_size=ARENA_SIZE, mountain_height=MOUNTAIN_HEIGHT):
    """Scatter spheres to form a Gaussian mountain."""
    def gaussian(x, y, sigma=arena_size/4):
        return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

    for _ in range(num_rocks):
        x = random.uniform(-arena_size/2, arena_size/2)
        y = random.uniform(-arena_size/2, arena_size/2)
        z = gaussian(x, y)
        size = random.uniform(0.1, max_size) * (1 - z / mountain_height)
        orientation = p.getQuaternionFromEuler([random.uniform(0, math.pi) for _ in range(3)])
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=[0.6, 0.4, 0.2, 1])
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=shape,
                          baseVisualShapeIndex=vis,
                          basePosition=[x, y, z + size],
                          baseOrientation=orientation)


def make_arena(arena_size=ARENA_SIZE, wall_height=1):
    """Create perimeter walls around the plane to contain creatures."""
    wall_h = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size/2, 0.5, wall_height/2])
    wall_v = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, arena_size/2, wall_height/2])
    for pos in [(0, arena_size/2, wall_height/2), (0, -arena_size/2, wall_height/2)]:
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=wall_h,
                          baseVisualShapeIndex=wall_h,
                          basePosition=pos)
    for pos in [(arena_size/2, 0, wall_height/2), (-arena_size/2, 0, wall_height/2)]:
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=wall_v,
                          baseVisualShapeIndex=wall_v,
                          basePosition=pos)

#  Fitness evaluation 
def evaluate_creature(cr, sim_steps=1200):
    """Run the creature from outside the mountain and measure climb + proximity."""
    spawn_x = -ARENA_SIZE/2 + SPAWN_MARGIN
    spawn_y = random.uniform(-SPAWN_MARGIN, SPAWN_MARGIN)
    yaw = random.uniform(0, 2 * math.pi)

    urdf_file = 'temp_creature.urdf'
    with open(urdf_file, 'w') as f:
        f.write(cr.to_xml())

    orientation = p.getQuaternionFromEuler([0, 0, yaw])
    robot_id = p.loadURDF(urdf_file, basePosition=[spawn_x, spawn_y, SPAWN_HEIGHT], baseOrientation=orientation)
    start_z = None
    max_z = -np.inf
    invalid = False

    for _ in range(sim_steps):
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        if start_z is None:
            start_z = pos[2]
        max_z = max(max_z, pos[2])
        if pos[2] > MOUNTAIN_HEIGHT + 2:
            invalid = True
            break
        time.sleep(1/240)

    p.removeBody(robot_id)
    if invalid or start_z is None:
        return 0
    vertical_gain = max_z - start_z
    xy_dist = math.hypot(pos[0], pos[1])
    proximity = max(0, (ARENA_SIZE/4) - xy_dist) / (ARENA_SIZE/4)
    return vertical_gain + proximity

#  GA evolution 
def evolve_with_config(pop, elite_fraction, mutation_rate):
    pop.creatures.sort(key=lambda c: c.fitness, reverse=True)
    elites = pop.creatures[:max(2, int(elite_fraction * len(pop.creatures)))]
    new_gen = elites.copy()
    while len(new_gen) < len(pop.creatures):
        p1, p2 = random.sample(elites, 2)
        child_genes = genome.crossover(p1.genes, p2.genes)
        genome.mutate(child_genes, mutation_rate)
        new_gen.append(creature.Creature(genes=child_genes))
    pop.creatures = new_gen

# Experiment configurations 
configs = [
    {"pop_size": 10, "gene_count": 3, "generations": 5, "mutation_rate": 0.05, "elite_fraction": 0.6},
    {"pop_size": 10, "gene_count": 3, "generations": 5, "mutation_rate": 0.05, "elite_fraction": 0.4},
    {"pop_size": 10, "gene_count": 3, "generations": 5, "mutation_rate": 0.05, "elite_fraction": 0.2},
]

# Environment
make_arena()
make_mountain(num_rocks=300, max_size=0.5)

# Run experiments
results = {}
for cfg in configs:
    print(f"\n🔧 Running config: {cfg}")
    pop = population.Population(pop_size=cfg["pop_size"], gene_count=cfg["gene_count"])
    bests, avgs = [], []

    for gen in range(cfg["generations"]):
        fitnesses = [evaluate_creature(cr) for cr in pop.creatures]
        for cr, fit in zip(pop.creatures, fitnesses):
            cr.fitness = fit
        best = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        bests.append(best)
        avgs.append(avg)
        print(f"Generation {gen} best fitness: {best:.3f}, avg fitness: {avg:.3f}")
        evolve_with_config(pop, cfg["elite_fraction"], cfg["mutation_rate"])

    results[(cfg["pop_size"], cfg["mutation_rate"], cfg["elite_fraction"])] = (bests, avgs)
    print(f"📊 Bests: {bests}\n   Avgs:  {avgs}")

#  Save results 
import csv
with open("elite_fraction_experiment_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["pop_size", "gene_count", "mutation_rate", "elite_fraction"] + [f"gen_{i}_best" for i in range(cfg["generations"])] + [f"gen_{i}_avg" for i in range(cfg["generations"])]
    writer.writerow(header)
    for (ps, mr, ef), (bests, avgs) in results.items():
        row = [ps, cfg["gene_count"], mr, ef] + bests + avgs
        writer.writerow(row)

print("Experiments complete. Results written to elite_fraction experiment_results.csv")
