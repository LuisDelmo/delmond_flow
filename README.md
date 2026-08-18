# Delmond Flow

**Delmond Flow** is an experimental neural network framework built from scratch in Python and NumPy.

The goal of the project is not to replace frameworks such as TensorFlow or PyTorch, but to understand what happens underneath them by implementing the core components of a neural network manually: neurons, layers, activations, loss functions, forward propagation, backpropagation, and eventually optimization and GPU acceleration.

Instead of only using:

```python
model = Sequential([
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
```

the idea is to understand **what every line behind an API like this actually has to do**.

---

## Why Delmond Flow?

Modern deep learning libraries hide an enormous amount of complexity behind simple APIs.

Delmond Flow is my attempt to build those abstractions myself while learning the mathematics and software architecture behind neural networks.

The project explores concepts such as:

* matrix-based neural network computation
* neurons and dense layers
* weights and biases
* activation functions
* loss functions
* forward propagation
* backpropagation
* gradient descent
* model/layer abstractions
* vectorized NumPy operations
* eventually GPU-accelerated computation

The long-term goal is to create a small Keras-like API while keeping the internal implementation understandable.

---

# Current Architecture

```text
delmond_flow/
│
├── model/
│   ├── activations/
│   │   ├── base.py
│   │   ├── relu.py
│   │   └── sigmoid.py
│   │
│   ├── layers/
│   │   ├── base.py
│   │   └── dense.py
│   │
│   ├── base.py
│   ├── neuron.py
│   ├── sequential.py
│   └── __init__.py
│
├── loss/
│   ├── base.py
│   ├── meansquaredrrror.py
│   ├── sparcecrosscategoryentropy.py
│   └── __init__.py
│
├── delflow.py
└── main.py
```

---

# Core Concepts

## Neuron

The `Neuron` class represents the fundamental computational unit of the network.

Conceptually, each neuron computes:

[
z = XW + b
]

and then applies an activation function:

[
a = g(z)
]

A neuron stores information such as:

* inputs
* weights
* bias
* activation function
* output
* parent neurons/layers

---

## Layers

Layers organize multiple neurons into a single computational unit.

The basic layer abstraction is responsible for concepts such as:

```python
Layer(
    units=5,
    activation="relu"
)
```

A layer manages:

* number of units
* activation function
* input values
* weight matrices
* bias values
* neuron creation
* layer output

---

## Dense Layer

`Dense` extends the base `Layer` abstraction.

Example:

```python
from model.layers import Dense

layer = Dense(
    units=5,
    activation="relu"
)
```

Dense layers will perform the standard neural-network transformation:

[
Z = XW^T + b
]

followed by an activation function.

The implementation is being designed around **matrix operations rather than Python loops whenever possible**, allowing the framework to take advantage of NumPy's optimized numerical operations.

---

# Sequential Models

Delmond Flow contains a `Sequential` model abstraction inspired by APIs such as Keras.

Example of the intended interface:

```python
from model import Sequential
from model.layers import Dense

model = Sequential([
    Dense(5, activation="relu"),
    Dense(1, activation="linear")
])
```

The idea is that information flows sequentially through each layer:

```text
Input
  │
  ▼
Dense Layer
  │
  ▼
Activation
  │
  ▼
Dense Layer
  │
  ▼
Prediction
```

Training support is currently under development.

---

# Activation Functions

## ReLU

Rectified Linear Unit:

[
ReLU(z) = \max(0,z)
]

Used primarily in hidden layers.

Conceptually:

```python
output = np.maximum(0, z)
```

---

## Sigmoid

Sigmoid maps a value into the interval:

[
0 < \sigma(z) < 1
]

with:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

It is commonly useful for binary-output probabilities.

---

# Loss Functions

Loss functions measure how far the model's predictions are from the expected values.

Delmond Flow currently contains abstractions for regression and classification losses.

## Mean Squared Error

For regression:

[
MSE =
\frac{1}{m}
\sum_{i=1}^{m}
(y_i-\hat{y}_i)^2
]

Example conceptually:

```python
loss = np.mean((y_train - y_hat) ** 2)
```

---

## Sparse Categorical Cross-Entropy

For multi-class classification where labels are represented as integer class indices.

Conceptually:

[
L = -\log(\hat{y}_{correct})
]

For example, if:

```python
y_train = [2, 0, 1]
```

each value identifies the correct class for its corresponding training example.

---

# Neural Network Computation

A major goal of Delmond Flow is to keep neural-network operations vectorized.

Instead of calculating each neuron individually:

```text
Neuron 1
Neuron 2
Neuron 3
Neuron 4
...
```

the framework can represent an entire layer using matrices.

For a batch:

\[
W =
\begin{bmatrix}
w_{11} & w_{12} & \dots \\
w_{21} & w_{22} & \dots
\end{bmatrix}
\]

the layer computes:

\[
Z = XW^T + b
\]

This allows all neurons and training examples to be processed together.

---

# Project Status

> **Delmond Flow is currently a work in progress.**

The project is being built while studying the fundamentals of neural networks, so APIs and internal architecture may change significantly.

Current focus:

```text
Model architecture
        ↓
Layer abstraction
        ↓
Vectorized forward propagation
        ↓
Loss calculation
        ↓
Backpropagation
        ↓
Gradient descent
        ↓
Training loop
        ↓
Optimization
        ↓
GPU acceleration
```

---

# Roadmap

* [x] Base model abstraction
* [x] Sequential model structure
* [x] Layer abstraction
* [x] Dense layer abstraction
* [x] Neuron abstraction
* [x] ReLU activation
* [x] Sigmoid activation
* [x] Mean Squared Error
* [x] Sparse Categorical Cross-Entropy
* [ ] Complete weight initialization
* [ ] Complete vectorized forward propagation
* [ ] Softmax activation
* [ ] Backpropagation
* [ ] Activation derivatives
* [ ] Loss derivatives
* [ ] Gradient descent
* [ ] Optimizer abstraction
* [ ] SGD optimizer
* [ ] Mini-batch training
* [ ] `model.compile()`
* [ ] Complete `model.fit()`
* [ ] `model.predict()`
* [ ] Metrics
* [ ] Model serialization
* [ ] Tests
* [ ] Benchmarks
* [ ] GPU backend
* [ ] CUDA experimentation

---

# Installation

Clone the repository:

```bash
git clone https://github.com/LuisDelmo/delmond_flow.git
cd delmond_flow
```

Install NumPy:

```bash
pip install numpy
```

At the moment, the project is intended primarily for development and experimentation rather than production installation.

---

# Example

The intended high-level API is moving toward something similar to:

```python
from model import Sequential
from model.layers import Dense

model = Sequential([
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="linear"),
])

model.fit(
    X_train,
    y_train,
    epochs=100
)
```

As development continues, this API will evolve together with the underlying training engine.

---

# What I'm Learning Through This Project

Delmond Flow is primarily an educational engineering project.

Building a neural network framework from scratch provides a deeper understanding of topics that high-level frameworks normally hide:

### Linear algebra

```text
X @ W + b
```

Matrix multiplication, shapes, transposes, broadcasting, and vectorization.

### Calculus

```text
∂J/∂W
∂J/∂b
∂J/∂A
∂J/∂Z
```

The derivatives required by backpropagation.

### Neural networks

```text
Forward propagation
        ↓
Loss
        ↓
Backward propagation
        ↓
Gradients
        ↓
Parameter update
```

### Software engineering

Designing reusable abstractions such as:

```text
Model
├── Sequential
│
Layer
├── Dense
│
Activation
├── ReLU
└── Sigmoid
│
Loss
├── MSE
└── Sparse Categorical Cross-Entropy
```

---

# Philosophy

Delmond Flow follows one main idea:

> **Don't just use the abstraction — understand what the abstraction is hiding.**

Frameworks such as TensorFlow and PyTorch make neural networks easy to use.

This project explores what has to exist underneath that simplicity.

---

# Disclaimer

Delmond Flow is an experimental and educational project.

It is **not currently intended for production machine-learning workloads** and should not be considered an alternative to mature frameworks such as TensorFlow, PyTorch, or JAX.

The purpose is learning, experimentation, and building a neural-network engine from first principles.

---

# Author

**Luis Fernando Faria Delmondes**

Built while studying machine learning, neural networks, numerical computing, and the mathematics behind deep learning.
