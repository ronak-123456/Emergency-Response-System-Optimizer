# Emergency Response System Optimizer

## Overview

This Python program simulates and optimizes the placement of emergency service stations (fire, police, and medical) across a city to minimize response times. It uses advanced algorithms and data structures to create a realistic city model and find the optimal distribution of emergency services.

## Features

- Generates a simulated city with population density and road networks
- Uses genetic algorithms to optimize emergency station placement
- Calculates response times using Voronoi diagrams and weighted distances
- Visualizes results using matplotlib
- Utilizes parallel processing for improved performance

## Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- DEAP (Distributed Evolutionary Algorithms in Python)
- matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ronak-123456/emergency-response-optimizer.git
   ```

2. Install the required packages:
   ```
   pip install numpy scipy scikit-learn deap matplotlib
   ```

## Usage

Run the main script:

```
python emergency_response_optimizer.py
```

The program will generate a simulated city, optimize the placement of emergency stations, and save a visualization of the result as `optimized_placement.png`.

## Customization

You can modify the following parameters in the `main()` function to customize the simulation:

- `city_size`: The size of the simulated city grid
- `num_stations`: The number of emergency stations to place
- `population_size`: The size of the population for the genetic algorithm
- `num_generations`: The number of generations for the genetic algorithm

## How it Works

1. **City Generation**: Creates a simulated city with a population density map and road network.
2. **Optimization**: Uses a genetic algorithm to find the best placement of emergency stations.
3. **Response Time Calculation**: Employs Voronoi diagrams to partition the city and calculate weighted response times.
4. **Visualization**: Generates a heatmap of the city with optimized station locations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
