import random, math


def ga_tsp(
    num_cities: int = 14,
    pop_size: int = 120,
    generations: int = 600,
    mutation_rate: float = 0.15,
) -> float:
    """Evolves a TSP tour and returns the best tour length found.

    GA choices (initial seed, crossover order, mutation) create stochasticity.
    """
    # --- helpers -----------------------------------------------------------
    coords = [(random.random(), random.random()) for _ in range(num_cities)]

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def tour_len(t):
        return sum(
            dist(coords[t[i]], coords[t[(i + 1) % num_cities]])
            for i in range(num_cities)
        )

    def crossover(p1, p2):
        a, b = sorted(random.sample(range(num_cities), 2))
        child = [-1] * num_cities
        child[a:b] = p1[a:b]
        fill = [c for c in p2 if c not in child]
        ptr = 0
        for i in range(num_cities):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return child

    # --- GA loop -----------------------------------------------------------
    pop = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]
    best = min(pop, key=tour_len)
    for _ in range(generations):
        pop.sort(key=tour_len)
        next_pop = pop[: pop_size // 2]  # elitism
        while len(next_pop) < pop_size:
            p1, p2 = random.sample(pop[:60], 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                i, j = random.sample(range(num_cities), 2)
                child[i], child[j] = child[j], child[i]
            next_pop.append(child)
        pop = next_pop
        cand = min(pop, key=tour_len)
        if tour_len(cand) < tour_len(best):
            best = cand
    return tour_len(best)  # numeric: path length
