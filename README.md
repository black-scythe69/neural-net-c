# neural-net-c
# Neural Network in C from Scratch

> A feedforward neural network built entirely in C — no ML libraries, 
> no NumPy, just raw C and math.

## Why I built this
I wanted to understand how neural networks actually work under the hood,
not just call model.fit(). Every matrix operation, every gradient,
every weight update — implemented manually in C.

Built as an independent project in my 2nd semester of B.Tech Cybersecurity.

## Architecture
- Custom matrix library with heap-allocated flat arrays
- 2-2-1 fully connected network (2 input, 2 hidden, 1 output)
- Sigmoid activation function
- Mean squared error loss
- Backpropagation with stochastic gradient descent
- lr=2.0, taken intentionally for this toy network

## Loss Curve
- Epoch 0    -> 0.319910
- Epoch 100  -> 0.278194
- Epoch 200  -> 0.191175
- Epoch 300  -> 0.031606
- Epoch 400  -> 0.006548
- Epoch 500  -> 0.003460
- Epoch 600  -> 0.002319
- Epoch 700  -> 0.001733
- Epoch 6500 -> 0.000101

## Final Output (XOR)
- [0,0] -> 0.010069  (expected 0)
- [0,1] -> 0.988533  (expected 1)
- [1,0] -> 0.990609  (expected 1)
- [1,1] -> 0.008869  (expected 0)

## What I learned
Implementing matrix operations from scratch in C
Forward pass and backpropagation math
Sigmoid activation and its derivative
Gradient flow through layers
Why weight initialization matters for convergence
Debugging segfaults and memory issues in C
Early stopping at 14000 epochs, avg loss->0.001008

## Future improvements
Xavier weight initialization
Configurable network architecture
More hidden layers
Additional activation functions

## Compile and Run
```bash
gcc NeuralNet.c -o neural -lm
./neural