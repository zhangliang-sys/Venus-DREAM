# PRO-DREAM

A Deep Retrieval-Enhanced Meta-learning Framework for Enzyme Optimum pH Prediction

## Introduction

This is a meta-learning framework for predicting protein pH using MAML and Reptile algorithms.

## Key Features

- **Meta-Learning Implementations**:
  - **MAML** (Model-Agnostic Meta-Learning)
  - **Reptile** algorithm
  
- **Sequence Retrieval Strategies**:
  - **Similarity-based** retrieval using ESM2 embeddings
  - **Random** sequence selection
  - **Fixed random** selection
  - **Scaled retrieval** with different dataset sizes

- **Model Architecture**:
  - Based on **ESM2** protein language model
  - Supports both pretrained and randomly initialized models

## Installation

To install the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/protein-ph-prediction.git
cd protein-ph-prediction

# Install requirements
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare sequence retrieval data using different strategies:

```bash
python retrieval.py \
    --opt_train data/phopt_training.fasta \
    --opt_test data/phopt_testing.fasta \
    --opt_valid data/phopt_validation.fasta \
    --strategy opt_retrieval \
    --topk 5
```

### Model Training

#### MAML Training

Train the model using Model-Agnostic Meta-Learning:

```bash
python maml.py \
    --mode train \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5
```

#### Reptile Training

Train the model using the Reptile algorithm:

```bash
python reptile.py \
    --mode train \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5
```

### Model Evaluation

Evaluate trained models on the test set:

- **For MAML**:

```bash
python maml.py --mode test --retrieval_strategy opt_retrieval
```

- **For Reptile**:

```bash
python reptile.py --mode test --retrieval_strategy opt_retrieval
```

### Command Line Arguments

- **Mode Options**:
  - `--mode`: train or test
  - `--pretrained`: Use pretrained ESM2 model

- **Retrieval Strategy**:
  - `--retrieval_strategy`:
    - `opt_retrieval`: Similarity-based
    - `opt_random`: Random selection
    - `opt_fixed_random`: Fixed random selection
    - `opt_retrieval_scaled_0.2`: 20% scaled retrieval
    - `opt_retrieval_scaled_0.6`: 60% scaled retrieval

- **Training Parameters**:
  - `--topk`: Number of sequences to retrieve (default: 5)
  - `--meta_lr`: Meta-learning rate
  - `--inner_lr`: Inner loop learning rate
  - `--num_epochs`: Number of training epochs

## Project Structure

- **Core Components**:
  - `maml.py`: MAML implementation
  - `reptile.py`: Reptile implementation
  - `retrieval.py`: Sequence retrieval strategies

- **Supporting Files**:
  - `models/`: Model architectures
  - `utils/`: Utility functions
  - `data/`: Data directory

## Citation

Please cite our work if you use this code in your research:

```bibtex
@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.