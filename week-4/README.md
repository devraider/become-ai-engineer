# Week 4 - Deep Learning with PyTorch

Master the fundamentals of deep learning using PyTorch. Learn tensors, autograd, neural networks, and training loops to build and train models from scratch.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch Book (Free)](https://pytorch.org/deep-learning-with-pytorch)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

### Video Tutorials

- [PyTorch for Deep Learning - freeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA)
- [Neural Networks Explained - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

## Installation

```bash
cd week-4
uv init
uv venv
source .venv/bin/activate
uv add torch torchvision numpy pandas matplotlib scikit-learn pytest
```

> 💡 **Note**: For Apple Silicon (M1/M2/M3), PyTorch uses MPS acceleration automatically.

---

## Concepts

### 1. Why PyTorch?

PyTorch is the most popular deep learning framework for research and increasingly for production.

**PyTorch vs scikit-learn:**

| scikit-learn | PyTorch |
|--------------|---------|
| Traditional ML algorithms | Neural networks |
| CPU only | GPU/TPU acceleration |
| Fixed architectures | Custom architectures |
| Structured data | Any data (images, text, audio) |

**Key advantages:**
- **Dynamic computation graphs** - Debug like regular Python
- **Pythonic** - Feels natural, not like a separate DSL
- **Research to production** - Same code works everywhere
- **Huge ecosystem** - HuggingFace, Lightning, torchvision

---

### 2. Tensors - The Foundation

Tensors are multi-dimensional arrays, like NumPy arrays but with GPU support.

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])                    # From list
x = torch.zeros(3, 4)                          # 3x4 zeros
x = torch.ones(2, 3)                           # 2x3 ones
x = torch.randn(2, 3)                          # Random normal
x = torch.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]

# From NumPy
import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
back_to_numpy = tensor.numpy()

# Tensor properties
print(x.shape)      # Size of each dimension
print(x.dtype)      # Data type (float32, int64, etc.)
print(x.device)     # CPU or CUDA
```

**Tensor operations:**

```python
# Math operations (element-wise)
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

c = a + b           # Addition
c = a * b           # Multiplication
c = a ** 2          # Power
c = torch.sqrt(a)   # Square root

# Matrix operations
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B           # Matrix multiplication (2x4)
C = torch.matmul(A, B)  # Same thing

# Reshaping
x = torch.randn(12)
x = x.view(3, 4)          # Reshape to 3x4
x = x.reshape(2, 6)       # Another reshape
x = x.unsqueeze(0)        # Add dimension: (1, 2, 6)
x = x.squeeze()           # Remove size-1 dimensions
```

📝 **Exercise 1 (Basic)** - Practice tensors in [exercises/exercise_basic_1_tensors.py](exercises/exercise_basic_1_tensors.py)

---

### 3. Autograd - Automatic Differentiation

Autograd automatically computes gradients - the heart of training neural networks.

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Forward pass - compute output
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# Backward pass - compute gradients
y.backward()

# dy/dx = 2x + 3 = 2(2) + 3 = 7
print(x.grad)  # tensor([7.])
```

**How neural networks learn:**

```
1. Forward pass: Input → Model → Prediction
2. Loss: Compare prediction to target
3. Backward pass: Compute gradients (how to adjust weights)
4. Update: Adjust weights to reduce loss
5. Repeat
```

**Practical example:**

```python
# Simple linear regression with autograd
X = torch.tensor([[1.], [2.], [3.], [4.]])
y = torch.tensor([[2.], [4.], [6.], [8.]])  # y = 2x

# Learnable parameters
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.1

for epoch in range(100):
    # Forward pass
    y_pred = X * w + b
    
    # Loss (MSE)
    loss = ((y_pred - y) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Update weights (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

print(f"Learned: w={w.item():.2f}, b={b.item():.2f}")  # ~2, ~0
```

---

### 4. Neural Networks with nn.Module

PyTorch provides `nn.Module` for building neural networks.

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Create model
model = SimpleNN(input_size=10, hidden_size=32, output_size=2)

# Forward pass
x = torch.randn(5, 10)  # Batch of 5 samples
output = model(x)       # Shape: (5, 2)
```

**Common layers:**

```python
# Linear (fully connected)
nn.Linear(in_features, out_features)

# Activation functions
nn.ReLU()           # Most common
nn.Sigmoid()        # Output in (0, 1)
nn.Tanh()           # Output in (-1, 1)
nn.Softmax(dim=1)   # Probabilities (sum to 1)

# Regularization
nn.Dropout(p=0.5)   # Randomly zero 50% of inputs
nn.BatchNorm1d(num_features)  # Normalize activations

# Sequential model (simpler syntax)
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 2)
)
```

📝 **Exercise 2 (Intermediate)** - Build neural networks in [exercises/exercise_intermediate_2_neural_networks.py](exercises/exercise_intermediate_2_neural_networks.py)

---

### 5. Training Loop

The standard PyTorch training pattern:

```python
import torch.optim as optim

# Model, loss, optimizer
model = SimpleNN(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Training mode
    
    for batch_x, batch_y in train_loader:
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(batch_x)
        
        # 3. Compute loss
        loss = criterion(outputs, batch_y)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
    
    # Validation
    model.eval()  # Evaluation mode (no dropout)
    with torch.no_grad():
        val_outputs = model(val_x)
        val_loss = criterion(val_outputs, val_y)
```

**Loss functions:**

| Task | Loss Function |
|------|---------------|
| Binary classification | `nn.BCEWithLogitsLoss()` |
| Multi-class classification | `nn.CrossEntropyLoss()` |
| Regression | `nn.MSELoss()` or `nn.L1Loss()` |

**Optimizers:**

| Optimizer | When to Use |
|-----------|-------------|
| `SGD` | Simple baseline |
| `Adam` | Good default, works for most cases |
| `AdamW` | Adam with weight decay (recommended) |

---

### 6. DataLoader and Datasets

Efficient data loading for training:

```python
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Create dataset and loader
dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,      # Shuffle for training
    num_workers=4      # Parallel data loading
)

# Iterate
for batch_x, batch_y in train_loader:
    # batch_x shape: (32, ...)
    pass
```

---

### 7. GPU Acceleration

Move tensors and models to GPU for faster training:

```python
# Check availability
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Move data to device
x = x.to(device)
y = y.to(device)

# In training loop
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    # ... training code
```

📝 **Exercise 3 (Advanced)** - Complete training pipelines in [exercises/exercise_advanced_3_training.py](exercises/exercise_advanced_3_training.py)

---

### 8. Saving and Loading Models

```python
# Save model weights
torch.save(model.state_dict(), 'model.pth')

# Load model weights
model = SimpleNN(784, 128, 10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model (not recommended)
torch.save(model, 'full_model.pth')
model = torch.load('full_model.pth')

# Save checkpoint (for resuming training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

---

## Weekly Project

Build a **Neural Network Classifier** from scratch that:

1. Loads a dataset (MNIST digits or custom)
2. Creates a custom Dataset and DataLoader
3. Builds a multi-layer neural network
4. Implements complete training loop with validation
5. Tracks and plots training/validation loss
6. Saves the best model checkpoint
7. Evaluates on test set with accuracy and confusion matrix

📝 **Project file:** [exercises/project_pipeline.py](exercises/project_pipeline.py)

📝 **Tests:** [exercises/tests/test_project_pipeline.py](exercises/tests/test_project_pipeline.py)

---

## Interview Questions

### Basic Level

1. **What is a tensor and how is it different from a NumPy array?**
   - Tensor is a multi-dimensional array like NumPy
   - Differences: GPU support, automatic differentiation, part of computation graph
   - Can convert between them: `torch.from_numpy()`, `.numpy()`

2. **What is autograd and why is it important?**
   - Automatic differentiation - computes gradients automatically
   - Essential for training: gradients tell us how to adjust weights
   - `requires_grad=True` enables tracking, `.backward()` computes gradients

3. **What is the difference between `model.train()` and `model.eval()`?**
   - `train()`: Enables dropout, batch norm uses batch statistics
   - `eval()`: Disables dropout, batch norm uses running statistics
   - Always switch modes appropriately for training vs inference

### Intermediate Level

4. **Explain the PyTorch training loop steps.**
   - Zero gradients (`optimizer.zero_grad()`)
   - Forward pass (compute predictions)
   - Compute loss
   - Backward pass (`loss.backward()`)
   - Update weights (`optimizer.step()`)

5. **What is the purpose of `torch.no_grad()`?**
   - Disables gradient computation
   - Saves memory and computation during inference
   - Use during validation/testing

6. **How do you handle overfitting in neural networks?**
   - Dropout layers
   - Early stopping
   - Weight decay (L2 regularization)
   - Data augmentation
   - Reduce model complexity

### Advanced Level

7. **What is the vanishing/exploding gradient problem?**
   - Gradients become very small or large during backpropagation
   - Deep networks struggle to train
   - Solutions: ReLU activation, batch normalization, residual connections, proper initialization

8. **Explain batch normalization and why it helps.**
   - Normalizes layer inputs to zero mean, unit variance
   - Allows higher learning rates
   - Acts as regularization
   - Reduces internal covariate shift

9. **How would you debug a neural network that's not learning?**
   - Check data: correct labels, proper normalization
   - Start simple: single layer, small data
   - Verify gradients are flowing (not zero/NaN)
   - Try different learning rates
   - Check loss function matches task
   - Overfit on small batch first

---

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Create and manipulate PyTorch tensors
- [ ] Explain autograd and compute gradients manually
- [ ] Build neural networks using nn.Module
- [ ] Implement a complete training loop
- [ ] Use DataLoader for efficient batching
- [ ] Move models and data to GPU/MPS
- [ ] Save and load model checkpoints
- [ ] Choose appropriate loss functions and optimizers
- [ ] Debug common training issues
- [ ] Explain key concepts in interviews

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Continue to Week 5](../week-5/README.md)**
