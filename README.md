# PRO-DREAM
A Deep Retrieval-Enhanced Meta-learning Framework for Enzyme optimum pH Prediction
# Protein pH Prediction with Meta-Learning

**A meta-learning framework for predicting protein pH using MAML and Reptile algorithms**

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

```bash
# Clone the repository
git clone https://github.com/yourusername/protein-ph-prediction.git
cd protein-ph-prediction
```
# Install requirements
```bash
pip install -r requirements.txt
```

## Usage
### Data Preparation
Description: Prepare sequence retrieval data using different strategies
###Key Features:
Multiple retrieval options
Configurable top-k selection
Support for train/test/validation splits
```bash
python retrieval.py \
    --opt_train data/phopt_training.fasta \
    --opt_test data/phopt_testing.fasta \
    --opt_valid data/phopt_validation.fasta \
    --strategy opt_retrieval \
    --topk 5
```
### Model Training
MAML Training
Description: Train model using Model-Agnostic Meta-Learning
Key Parameters:
Meta learning rate: 0.0001
Inner learning rate: 0.0005
Number of epochs: 50
```bash
python maml.py \
    --mode train \
    --meta_lr 0.0001 \
    --inner_lr 0.0005 \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5
```
### Reptile Training
Description: Train model using Reptile algorithm
Key Parameters:
Meta learning rate: 1
Inner learning rate: 0.001
Number of epochs: 50
```bash
python reptile.py \
    --mode train \
    --meta_lr 1 \
    --inner_lr 0.001 \
    --num_epochs 50 \
    --retrieval_strategy opt_retrieval \
    --topk 5
```
### Model Evaluation
Description: Evaluate trained models on test set
Features:
Support for both MAML and Reptile
Multiple evaluation metrics
Prediction output in CSV format
# For MAML
python maml.py --mode test --retrieval_strategy opt_retrieval

# For Reptile
python reptile.py --mode test --retrieval_strategy opt_retrieval
Command Line Arguments
Mode Options:

--mode: train or test
--pretrained: Use pretrained ESM2 model
Retrieval Strategy:

--retrieval_strategy:
opt_retrieval: Similarity-based
opt_random: Random selection
opt_fixed_random: Fixed random selection
opt_retrieval_scaled_0.2: 20% scaled retrieval
opt_retrieval_scaled_0.6: 60% scaled retrieval
Training Parameters:

--topk: Number of sequences to retrieve (default: 5)
--meta_lr: Meta-learning rate
--inner_lr: Inner loop learning rate
--num_epochs: Number of training epochs
Project Structure
Core Components:

maml.py: MAML implementation
reptile.py: Reptile implementation
retrieval.py: Sequence retrieval strategies
Supporting Files:

models/: Model architectures
utils/: Utility functions
data/: Data directory
Citation
Please cite our work if you use this code in your research:

@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.