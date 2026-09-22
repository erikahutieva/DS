import random
import matplotlib.pyplot as plt

class Animal:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0

    def calculate_fitness(self, target):
        self.fitness = sum(1 for a, b in zip(self.genome, target) if a == b)
        return self.fitness

    def mutate(self, mutation_rate, gene_pool):
        genome_list = list(self.genome)
        for i in range(len(genome_list)):
            if random.random() < mutation_rate:
                genome_list[i] = random.choice(gene_pool)
        self.genome = ''.join(genome_list)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1.genome) - 1)
    child_genome = parent1.genome[:point] + parent2.genome[point:]
    return Animal(child_genome)

def select(population):
    tournament = random.sample(population, 3)
    tournament.sort(key=lambda x: x.fitness, reverse=True)
    return tournament[0]

def evolve(target, gene_pool, gene_length, population_size, mutation_rate, generations):
    population = [Animal(''.join(random.choice(gene_pool) for _ in range(gene_length))) for _ in range(population_size)]
    
    max_fitness_history = []
    avg_fitness_history = []
    min_fitness_history = []

    for gen in range(generations):
        for animal in population:
            animal.calculate_fitness(target)
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        max_fitness = population[0].fitness
        avg_fitness = sum(a.fitness for a in population) / population_size
        min_fitness = population[-1].fitness
        
        max_fitness_history.append(max_fitness)
        avg_fitness_history.append(avg_fitness)
        min_fitness_history.append(min_fitness)
        
        print(f"Поколение {gen+1}: Макс фитнес = {max_fitness}, Средний фитнес = {avg_fitness:.2f}, Мин фитнес = {min_fitness}")
        print(f"  Лучший геном: {population[0].genome}")
        
        if max_fitness == gene_length:
            print("Оптимальный геном достигнут!")
            break
        
        next_generation = population[:int(0.1*population_size)]
        
        while len(next_generation) < population_size:
            parent1 = select(population)
            parent2 = select(population)
            child = crossover(parent1, parent2)
            child.mutate(mutation_rate, gene_pool)
            next_generation.append(child)
        
        population = next_generation

    # Построение графиков
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(max_fitness_history)+1), max_fitness_history, label='Максимальный фитнес', marker='o')
    plt.plot(range(1, len(avg_fitness_history)+1), avg_fitness_history, label='Средний фитнес', marker='x')
    plt.plot(range(1, len(min_fitness_history)+1), min_fitness_history, label='Минимальный фитнес', marker='.')
    plt.title('Эволюция популяции по поколениям')
    plt.xlabel('Поколение')
    plt.ylabel('Фитнес (число совпадений)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    GENE_POOL = ['A', 'T', 'C', 'G']
    GENE_LENGTH = 20
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.01
    GENERATIONS = 100

    TARGET_GENOME = "ATCGTACGATCGTACGATCG"

    print(f"Целевой геном: {TARGET_GENOME}")
    evolve(TARGET_GENOME, GENE_POOL, GENE_LENGTH, POPULATION_SIZE, MUTATION_RATE, GENERATIONS)
