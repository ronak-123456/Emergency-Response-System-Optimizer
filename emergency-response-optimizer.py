import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans
from deap import base, creator, tools, algorithms
import multiprocessing
import random

class City:
    def __init__(self, size, population_density, road_network):
        self.size = size
        self.population_density = population_density
        self.road_network = road_network

class EmergencyStation:
    def __init__(self, x, y, station_type):
        self.x = x
        self.y = y
        self.type = station_type

def generate_city(size, complexity):
    population_density = np.random.exponential(scale=1.0, size=(size, size))
    population_density /= np.max(population_density)
    
    road_network = np.zeros((size, size))
    for _ in range(int(complexity * size)):
        x, y = np.random.randint(0, size, 2)
        direction = np.random.choice(['h', 'v'])
        if direction == 'h':
            road_network[x, :] = 1
        else:
            road_network[:, y] = 1
    
    return City(size, population_density, road_network)

def calculate_response_time(city, stations):
    voronoi = Voronoi([(s.x, s.y) for s in stations])
    response_times = np.zeros_like(city.population_density)
    
    for region, station in zip(voronoi.regions, stations):
        if -1 not in region and len(region) > 0:
            polygon = [voronoi.vertices[i] for i in region]
            for x in range(city.size):
                for y in range(city.size):
                    if point_in_polygon(x, y, polygon):
                        distance = ((x - station.x)**2 + (y - station.y)**2)**0.5
                        response_times[x, y] = distance / (1 + city.road_network[x, y])
    
    return np.average(response_times, weights=city.population_density)

def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def optimize_station_placement(city, num_stations, population_size, num_generations):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, city.size)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_stations * 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        stations = [EmergencyStation(individual[i], individual[i+1], 'general') for i in range(0, len(individual), 2)]
        return (calculate_response_time(city, stations),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=city.size/10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

    pool.close()
    return hof[0]

def visualize_result(city, best_individual):
    stations = [EmergencyStation(best_individual[i], best_individual[i+1], 'general') for i in range(0, len(best_individual), 2)]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(city.population_density, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Population Density')
    
    for x in range(city.size):
        for y in range(city.size):
            if city.road_network[x, y] == 1:
                plt.plot(y, x, 'k.', markersize=1)
    
    for station in stations:
        plt.plot(station.y, station.x, 'r*', markersize=10)
    
    plt.title('Optimized Emergency Station Placement')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig('optimized_placement.png')
    plt.close()

def main():
    city_size = 100
    num_stations = 10
    population_size = 100
    num_generations = 50

    city = generate_city(city_size, complexity=0.1)
    best_individual = optimize_station_placement(city, num_stations, population_size, num_generations)
    visualize_result(city, best_individual)

    print(f"Best fitness: {best_individual.fitness.values[0]}")
    print(f"Optimized station coordinates: {[(best_individual[i], best_individual[i+1]) for i in range(0, len(best_individual), 2)]}")

if __name__ == "__main__":
    main()
