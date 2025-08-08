# Warm Start Alpha Mining

A genetic programming system for discovering and enhancing quantitative trading factors (alphas) using warm start optimization techniques.

## 📄 Research Paper

This implementation is based on the research paper:

**["Alpha Mining and Enhancing via Warm Start Genetic Programming for Quantitative Investment"](https://arxiv.org/abs/2412.00896)**  
*Authors: Weizhe Ren, Yichen Qin, Yang Li*  
*Published: December 2024*

## 🎯 Overview

Traditional genetic programming often struggles in alpha factor discovery due to vast search spaces and computational burden. This project implements a novel "Warm Start Genetic Programming" (WSGP) approach that:

- **Starts from proven structures** rather than random initialization
- **Preserves effective alpha architectures** during evolution
- **Focuses search efforts** on promising regions of the solution space
- **Achieves 3x improvement** in effective alpha discovery rate vs traditional GP

## 🏗️ Architecture

### Core Components

- **`alpha_tree.py`**: Tree-based representation of mathematical alpha expressions
- **`warm_start_gp.py`**: Main genetic programming engine with warm start capabilities
- **`fitness.py`**: Information Coefficient (IC) based fitness evaluation
- **`data_loader.py`**: Yahoo Finance data integration and preprocessing
- **`evaluation/`**: Comprehensive backtesting and performance metrics

### Alpha Representation

Alphas are represented as expression trees supporting:

- **Mathematical Operations**: ADD, SUB, MUL, DIV
- **Statistical Functions**: MA (moving average), STD (standard deviation), ZSCORE
- **Ranking Operations**: ASC_RANK (cross-sectional ranking)
- **Temporal Operations**: DELAY (time lag)
- **Subset Operations**: TOP_N, BOTTOM_N, ABOVE_MEAN, BELOW_MEAN

## 🚀 Key Features

### Warm Start Genetic Programming
- **Structured Initialization**: Starts with effective alpha structures from research literature
- **Constrained Evolution**: Preserves tree structure while optimizing parameters and data fields
- **Dual-objective Fitness**: Balances training performance with validation stability

### Advanced Evaluation
- **Information Coefficient**: Both Pearson and Spearman correlations
- **Strategy Backtesting**: Full portfolio simulation with transaction costs
- **Risk Metrics**: Sharpe ratio, maximum drawdown, hit ratio analysis

## 📊 Performance Results

The system has discovered high-performing alphas including:

```
Best Alpha Structures:
- std(above_mean(+(returns,returns))): IC=0.319, IR=0.825
- std(top_n(*(returns,volume))): IC=0.218, IR=0.571  
- zscore(above_mean(/(returns,low))): IC=0.252, IR=0.720
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aandrewschwartzz/warm-start-alpha-mining.git
   cd warm-start-alpha-mining
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run experiments**:
   ```bash
   python src/run_experiment.py
   ```

## 📁 Project Structure

```
alpha_mining/
├── src/
│   ├── alpha_tree.py           # Alpha expression tree implementation
│   ├── warm_start_gp.py        # Genetic programming engine
│   ├── data_loader.py          # Data fetching and preprocessing
│   ├── fitness.py              # Fitness evaluation functions
│   ├── evaluate_alpha.py       # Alpha evaluation utilities
│   └── evaluation/             # Performance metrics and reporting
├── evaluation_results/         # Experimental results and metrics
├── 101Alphas.pdf              # Reference: Kakushadze alpha factors
├── alpha_mining.pdf           # Research paper PDF
└── requirements.txt           # Python dependencies
```

## 🔬 Research Contributions

1. **Novel Methodology**: First practical implementation of structure-preserving GP for alpha mining
2. **Empirical Validation**: Demonstrates 3x performance improvement over traditional GP
3. **Interpretability**: Maintains alpha interpretability crucial for risk management
4. **Production Ready**: Provides complete pipeline from discovery to backtesting

## 📈 Usage Example

```python
from src.warm_start_gp import WarmStartGP
from src.alpha_tree import AlphaTree
from src.data_loader import load_stock_data

# Load market data
data = load_stock_data(['AAPL', 'MSFT', 'GOOGL'])

# Initialize with a warm start alpha
warm_start_alpha = AlphaTree.from_string("std(above_mean(returns))")

# Create and run GP
gp = WarmStartGP(
    warm_start_alpha=warm_start_alpha,
    population_size=50,
    max_generations=100
)

best_alpha, best_fitness = gp.evolve(data)
print(f"Best alpha: {best_alpha}")
print(f"Information Coefficient: {best_fitness:.4f}")
```

## 📚 References

- Ren, W., Qin, Y., & Li, Y. (2024). "Alpha Mining and Enhancing via Warm Start Genetic Programming for Quantitative Investment." *arXiv:2412.00896*
- Kakushadze, Z. (2015). "101 Formulaic Alphas." *arXiv:1601.00991*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source. Please cite the original research paper if you use this code in your research.

## 🙏 Acknowledgments

- Original research by Weizhe Ren, Yichen Qin, and Yang Li
- Inspired by the 101 Formulaic Alphas framework
- Built with Python scientific computing stack