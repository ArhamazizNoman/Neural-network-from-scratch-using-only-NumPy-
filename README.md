# Neural Network from Scratch using only NumPy

This project implements **fully-connected feedforward neural networks (FFN) from first principles using only NumPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch. The implementation emphasizes the mathematical foundations of deep learning and demonstrates a complete understanding of neural network fundamentals.

## Project Overview

This repository contains two main implementations for image classification tasks:

1. **CIFAR-10 Classification** (`CIFAR-10_final.ipynb`)
2. **Fashion-MNIST Classification** (`fashion_final.ipynb`)

Both implementations showcase:
- **Complete forward and backward propagation** with manual gradient computation
- **Multiple optimization algorithms**: SGD with Momentum, RMSProp, and Adam
- **Various activation functions**: ReLU, Sigmoid, Tanh, Softmax
- **Weight initialization strategies**: He, Xavier/Glorot, Random
- **L2 regularization** to prevent overfitting
- **Comprehensive experiment tracking** using Weights & Biases (WandB)
- **Systematic hyperparameter exploration** through automated sweeps

## Datasets

### CIFAR-10
- **Description**: 60,000 32×32 color images across 10 classes
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Input Dimension**: 3,072 features (32 × 32 × 3 when flattened)
- **Best Performance**: **52.15% test accuracy**

### Fashion-MNIST
- **Description**: 70,000 28×28 grayscale images of fashion items
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Input Dimension**: 784 features (28 × 28 when flattened)
- **Best Performance**: **~89.3% test accuracy**

## Architecture

### Key Components

#### 1. **Activation Functions**
- **ReLU** (Rectified Linear Unit): Primary activation for hidden layers
- **Softmax**: Output layer for multi-class classification
- **Sigmoid & Tanh**: Implemented for completeness and comparison

#### 2. **Loss Functions**
- **Categorical Cross-Entropy**: Primary loss function for classification
- **Mean Squared Error**: Implemented for regression tasks
- **L2 Regularization**: Weight decay to prevent overfitting

#### 3. **Optimizers**
- **SGD with Momentum**: Stochastic gradient descent with velocity accumulation
- **RMSProp**: Adaptive learning rates using moving average of squared gradients
- **Adam**: Adaptive moment estimation combining momentum and RMSProp (primary optimizer)

#### 4. **Weight Initialization**
- **He Initialization**: Optimized for ReLU activations
- **Xavier/Glorot Initialization**: For Sigmoid/Tanh activations
- **Random Initialization**: Baseline comparison

#### 5. **Network Architecture**
- **Modular Layer Design**: Each layer handles forward/backward propagation independently
- **Fully Vectorized**: All operations use NumPy matrix operations (no loops over samples)
- **Configurable Depth**: Supports any number of hidden layers with configurable widths

## Results

### CIFAR-10 Results

**Best Configuration:**
- **Architecture**: [3072, 512, 256, 10] (820,874 parameters)
- **Weight Initialization**: He initialization
- **Optimizer**: Adam (learning_rate=0.001, β1=0.9, β2=0.999)
- **Batch Size**: 128
- **L2 Regularization**: λ = 0.001
- **Epochs**: 50

**Performance:**
- **Test Accuracy**: **52.15%**
- **Training Accuracy**: 57.67%
- **Validation Accuracy**: 51.96%
- **Train-Val Gap**: 5.71% (moderate overfitting)

**Hyperparameter Sweep Results:**
- Conducted **10 sweep runs** using WandB Bayesian optimization
- Performance range: **10.00% - 52.15%** (demonstrating hyperparameter sensitivity)
- **Key Findings**:
  - He initialization critical: ~20% performance difference vs Random initialization
  - Adam optimizer most robust: outperformed SGD and RMSProp
  - Baseline configuration validated as near-optimal through systematic exploration

### Fashion-MNIST Results

**Best Configuration:**
- **Architecture**: [784, 128, 128, 128, 10] (134,794 parameters)
- **Weight Initialization**: He initialization
- **Optimizer**: Adam (learning_rate=0.001)
- **Batch Size**: 64
- **L2 Regularization**: λ = 0.0001
- **Epochs**: 20

**Performance:**
- **Test Accuracy**: **~89.3%**
- **Training Accuracy**: ~91.7%
- **Validation Accuracy**: ~88.4%
- **Train-Val Gap**: ~2% (excellent generalization)

**Class-wise Performance:**
- **Best Classes**: Trouser (98.5%), Bag (97.4%), Sandal (95.3%)
- **Challenging Classes**: Shirt (72.1%), Pullover (79.7%)

## Key Technical Insights

### 1. Weight Initialization Impact
- **He initialization** is crucial for ReLU networks: Achieved 50–52% on CIFAR-10 vs 32–50% with random initialization
- Proper initialization prevents vanishing/exploding gradients and enables stable training
- Initialization strategy interacts significantly with optimizer choice

### 2. Optimizer Comparison
- **Adam optimizer** consistently outperformed alternatives:
  - Combines momentum acceleration with per-parameter adaptive learning rates
  - Most robust across different hyperparameter configurations
  - Requires minimal manual tuning
- **RMSProp** showed high sensitivity to initialization and learning rate
- **SGD with Momentum** performed competitively but required more careful tuning

### 3. Regularization Effects
- **L2 Regularization** (λ = 0.001–0.0001) effectively prevented overfitting
- Optimal regularization strength balances model capacity and generalization
- Too much regularization (λ > 0.006) caused severe underfitting

### 4. Architecture Design
- **Depth vs Width**: 2–3 hidden layers with 128–512 units provided optimal balance
- Deeper networks (4+ layers) showed diminishing returns with fully-connected architecture
- Architecture choice significantly impacts parameter count and training time

### 5. Training Dynamics
- **Convergence Patterns**: Smooth loss decrease indicates stable training
- **Gradient Flow**: Healthy gradient magnitudes (no vanishing/exploding) with proper initialization
- **Overfitting Monitoring**: Train-validation gap provides early warning of overfitting

## Implementation Highlights

### Pure NumPy Implementation
- **No Framework Dependencies**: Built entirely with NumPy arrays and matrix operations
- **Fully Vectorized**: All operations use batch processing (no loops over samples)
- **Memory Efficient**: Caches intermediate values for efficient backward propagation

### Modular Design
- **Separate Classes**: Layers, activations, optimizers, and loss functions are independently implementable
- **Easy Extensibility**: Simple to add new activation functions, optimizers, or loss functions
- **Clean Architecture**: Clear separation of concerns enables easy debugging and modification

### Production-Ready Features
- **Error Handling**: Proper handling of edge cases and numerical stability
- **Logging**: Comprehensive logging of training progress and metrics
- **Visualization**: Built-in plotting of training curves, confusion matrices, and weight distributions
- **Experiment Tracking**: Full WandB integration for experiment management

## Code Structure

### CIFAR-10 Implementation (`CIFAR-10_final.ipynb`)

1. **Data Pipeline** (`DataLoader` class)
   - Loads CIFAR-10 batch files
   - Handles image format conversion and normalization
   - Creates mini-batches for training

2. **Activation Functions** (`Activation` base class)
   - ReLU, Softmax, Sigmoid, Tanh implementations
   - Forward and backward methods

3. **Layer Implementation** (`Layer` class)
   - Fully-connected layer with configurable initialization
   - Forward and backward propagation
   - Gradient computation with L2 regularization

4. **Loss Functions** (`Loss` base class)
   - Categorical Cross-Entropy with L2 regularization
   - Mean Squared Error implementation

5. **Optimizers** (`Optimizer` base class)
   - SGD with Momentum
   - RMSProp
   - Adam

6. **Neural Network** (`FeedForwardNN` class)
   - Assembles layers into complete network
   - Training, prediction, and evaluation methods

7. **Training Infrastructure** (`Trainer` class)
   - Training loop with WandB integration
   - Mini-batch processing
   - Validation and metric tracking

8. **Hyperparameter Sweeps**
   - WandB Bayesian optimization
   - Systematic hyperparameter exploration

### Fashion-MNIST Implementation (`fashion_final.ipynb`)

Similar structure with Fashion-MNIST specific adaptations:
- CSV data loading
- 784-dimensional input handling
- Extended hyperparameter sweep configurations


## Getting Started

### Prerequisites

```bash
numpy
pandas
matplotlib
scikit-learn
wandb
seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ArhamazizNoman/Neural-network-from-scratch-using-only-NumPy-.git
cd Neural-network-from-scratch-using-only-NumPy-
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn wandb seaborn
```

3. Set up WandB (optional, for experiment tracking):
```bash
wandb login
```

4. Download datasets:
   - **CIFAR-10**: Place dataset in `Dataset_To_Use/cifar-10-batches-py/`
   - **Fashion-MNIST**: Place CSV files in `Fashion_Dataset/`

### Usage

1. **Run CIFAR-10 Classification**:
   - Open `CIFAR-10_final.ipynb`
   - Execute cells sequentially
   - Modify hyperparameters in Cell 18 for custom configurations

2. **Run Fashion-MNIST Classification**:
   - Open `fashion_final.ipynb`
   - Execute cells sequentially
   - Adjust hyperparameters in the configuration dictionary

3. **Run Hyperparameter Sweeps**:
   - Configure sweep parameters in the notebook
   - Execute sweep cells to launch automated experiments
   - View results in WandB dashboard

## Experiment Tracking with WandB

All experiments are logged to Weights & Biases for:
- **Real-time Monitoring**: Track training progress and metrics
- **Hyperparameter Comparison**: Compare different configurations side-by-side
- **Visualization**: Automatic plotting of training curves
- **Reproducibility**: All hyperparameters automatically logged
- **Collaboration**: Share results with team members

### WandB Projects
- **CIFAR-10**: `cifar10-numpy-nn`
- **Fashion-MNIST**: `fashion-mnist-numpy`

## Key Learnings

### Understanding Neural Network Fundamentals
- **Forward Propagation**: How data flows through layers with matrix multiplications
- **Backward Propagation**: Chain rule application for gradient computation
- **Optimization**: Different strategies for updating weights
- **Regularization**: Techniques to prevent overfitting

### Hyperparameter Sensitivity
- **Systematic Exploration**: Hyperparameter sweeps revealed significant performance variations
- **Interactions**: Hyperparameters interact in complex ways (e.g., initialization + optimizer)
- **Optimal Configurations**: Baseline configurations validated through systematic exploration

### Training Dynamics
- **Convergence Patterns**: Understanding loss curves and accuracy improvements
- **Overfitting Detection**: Train-validation gap as indicator of generalization
- **Gradient Health**: Monitoring gradient norms for training stability

### Practical Implementation
- **Numerical Stability**: Techniques for stable computations (e.g., softmax with max subtraction)
- **Efficiency**: Vectorized operations for computational efficiency
- **Modularity**: Clean code structure for maintainability and extensibility

## Mathematical Foundations

### Forward Propagation
For layer $l$ with input $A^{[l-1]}$ (where $A^{[0]} = X$):

$$
Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}
$$
$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

- $g^{[l]}(\cdot)$ is the activation function of layer $l$ (e.g., ReLU, Tanh, etc.)
- For the output layer $L$, $g^{[L]}(\cdot)$ is Softmax

### Backward Propagation

#### Output Layer (Softmax + Categorical Cross-Entropy)
When using Softmax activation combined with cross-entropy loss, the gradient simplifies to:

$$
dZ^{[L]} = \hat{Y} - Y \quad \in \mathbb{R}^{m \times C}
$$

(where $\hat{Y}$ are the predicted probabilities and $Y$ is the one-hot true label matrix)

#### Hidden Layers
For $l = L-1, L-2, \dots, 1$:

$$
dA^{[l]} = dZ^{[l+1]} (W^{[l+1]})^T
$$
$$
dZ^{[l]} = dA^{[l]} \odot (g^{[l]})'(Z^{[l]})
$$
$$
dW^{[l]} = \frac{1}{m} (A^{[l-1]})^T dZ^{[l]} + \lambda W^{[l]} \quad \text{(L2 regularization gradient)}
$$
$$
db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} (dZ^{[l]})^{(i)} \quad \text{(sum over batch dimension)}
$$

- $\odot$ denotes element-wise (Hadamard) product
- $(g^{[l]})'(\cdot)$ is the derivative of the activation function (e.g., ReLU' = 1 if $Z > 0$, else 0)

### Loss Function
Categorical Cross-Entropy Loss **with L2 Regularization**:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) 
+ \frac{\lambda}{2m} \sum_{l=1}^{L} \| W^{[l]} \|_F^2
$$

- First term: standard cross-entropy over $m$ training examples
- Second term: L2 weight decay (Frobenius norm) over all weight matrices
- Note: The regularization term is divided by $m$ (common in practice for scale-invariance w.r.t. batch size)

This formulation matches exactly what is implemented in modern frameworks (PyTorch, TensorFlow) and in your NumPy code.

## Educational Value

This project provides:
- **Complete Transparency**: Every operation is explicit and understandable
- **Mathematical Clarity**: Direct implementation of neural network mathematics
- **Framework Independence**: Understanding not tied to specific libraries
- **Experimentation Platform**: Easy to modify and test different configurations
- **Best Practices**: Professional code structure and experiment tracking

## Future Improvements

Potential enhancements:
- **Convolutional Layers**: Add CNN support for better image classification
- **Dropout Regularization**: Alternative regularization technique
- **Batch Normalization**: Improve training stability and convergence
- **Learning Rate Scheduling**: Adaptive learning rate decay
- **Early Stopping**: Prevent overfitting during training
- **Data Augmentation**: Improve generalization with image transformations

## Team

**Group 2 - Deep Learning 02456**
- DTU Business Analytics - Term 2

## License

This project is part of a Deep Learning course assignment at DTU (Technical University of Denmark).

## Acknowledgments

- **CIFAR-10 Dataset**: Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **Fashion-MNIST Dataset**: Created by Zalando Research
- **Weights & Biases**: For excellent experiment tracking platform
- **Course**: 02456 Deep Learning at DTU


---

**Note**: This implementation prioritizes educational value and understanding over performance optimization. For production use, frameworks like PyTorch or TensorFlow are recommended for their optimized operations and additional features.

