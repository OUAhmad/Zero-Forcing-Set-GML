# Graph Machine Learning for Zero Forcing Sets

This repository contains the **code and benchmark datasets** used in the paper:

**A Graph Machine Learning Framework to Compute Zero Forcing Sets in Graphs**  
*IEEE Transactions on Network Science and Engineering*

The project introduces a **graph machine learning framework** for computing **zero forcing sets (ZFS)** in graphs—an NP-hard combinatorial problem—using scalable, learning-based alternatives to classical greedy heuristics.

---

## Repository Structure

```
.
├── Code/
│   ├── modules/
│   │   ├── models.py              # GCN model definitions
│   │   ├── layers.py              # Custom neural network layers
│   │   ├── preprocessing.py       # Graph preprocessing utilities
│   │   ├── inits.py               # Model initialization routines
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── greedy.py              # Greedy ZFS baseline
│   │   ├── greedy_threading.py    # Parallelized greedy baseline
│   │   ├── size_est.py            # ZFS size estimation utilities
│   │   └── utils.py               # Helper functions
│   ├── reg_model/
│   │   ├── Regressor.joblib       # Trained ZFS size regressor
│   │   └── train.png              # Training loss visualization
│   ├── train.py                   # GCN training script
│   ├── Train_Regressor.py         # Regression model training script
│   ├── ZFS_Greedy.py              # Greedy ZFS benchmarking script
│   ├── Test.py                    # Model evaluation script
│   ├── sample.mat                 # Example graph file
│   └── Requirements.txt           # Python dependencies
│
├── Data/
│   ├── small_ER/                  # Small ER graphs (with optimal ZFS)
│   ├── Large_Graph/               # Large benchmark graphs
│   └── ...                        # Additional datasets
│
└── README.md
```

---

## Data Description

The `Data/` directory contains multiple subdirectories with **MATLAB `.mat` files** representing graph instances and associated zero forcing information.

- Each `.mat` file contains the graph adjacency matrix and one or more zero forcing set realizations.
- For graphs in `Data/small_ER/`, **optimal zero forcing sets** are available and were computed using a wavefront-based algorithm.
- Larger graphs are used for scalability evaluation and benchmarking against greedy heuristics.

These datasets are used for:
- Supervised training of learning-based models  
- Benchmarking against greedy and randomized heuristics  
- Evaluating generalization across different graph families  

---

## Methods Implemented

- **Graph Convolutional Network (GCN)**–based framework for ZFS computation  
- Classical **greedy and randomized greedy** baselines  
- **Parallelized greedy algorithm** for large graphs  
- **Regression-based estimation** of ZFS size using graph structural features  
- End-to-end **training, validation, and benchmarking pipelines**  

---

## Installation

```
pip install -r Code/Requirements.txt
```

Tested with **Python 3.7**.

---

## Usage

### Train the GCN model
```
python Code/train.py
```

### Train the ZFS size regressor
```
python Code/Train_Regressor.py
```

### Run greedy ZFS baseline on large graphs
```
python Code/ZFS_Greedy.py
```

### Evaluate trained models
```
python Code/Test.py
```

---

## Reproducibility

- Pretrained models are included (`Regressor.joblib`)
- Benchmark graphs with optimal and heuristic solutions are provided
- All experiments reported in the paper can be reproduced using the provided scripts

---

## Citation

```
@article{ahmad2024graph,
  title={A Graph Machine Learning Framework to Compute Zero Forcing Sets in Graphs},
  author={Ahmad, Obaid Ullah and Shabbir, Mudassir and Abbas, Waseem and Koutsoukos, Xenofon},
  journal={IEEE Transactions on Network Science and Engineering},
  year={2024}
}
```

---

## License

This repository is intended for **research and academic use only**.  
Please contact the authors for redistribution or commercial use.
