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

# Install requirements
pip install -r requirements.txt