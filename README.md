# Warm Start Alpha Mining

A genetic programming system for discovering and enhancing quantitative trading factors (alphas) using warm start optimization techniques.

## Research Paper

This implementation is based on the research paper. Gracious thanks to Ren, Qin, and Li:

**["Alpha Mining and Enhancing via Warm Start Genetic Programming for Quantitative Investment"](https://arxiv.org/abs/2412.00896)**  

## Overview

Traditional genetic programming often struggles in alpha factor discovery due to vast search spaces and computational burden. This project implements a novel "Warm Start Genetic Programming" (WSGP) approach that:

- **Starts from proven structures** rather than random initialization
- **Preserves effective alpha architectures** during evolution
- **Focuses search efforts** on promising regions of the solution space
- **Achieves 3x improvement** in effective alpha discovery rate vs traditional GP

## Architecture

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
