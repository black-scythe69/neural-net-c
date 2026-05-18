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

## Loss Curve
Epoch 0     -> 0.345183
Epoch 1000  -> 0.015598
Epoch 2500  -> 0.004466
Epoch 5000  -> 0.001964
Epoch 10000 -> 0.000909

## Final Output (XOR)
[0,0] -> 0.030384  (expected 0)
[0,1] -> 0.992054  (expected 1)
[1,0] -> 0.957601  (expected 1)
[1,1] -> 0.028842  (expected 0)

## What I learned
Implementing matrix operations from scratch in C
Forward pass and backpropagation math
Sigmoid activation and its derivative
Gradient flow through layers
Why weight initialization matters for convergence
Debugging segfaults and memory issues in C
Early stopping at 9000 epochs, avg loss->0.001019

## Future improvements
Xavier weight initialization
Configurable network architecture
More hidden layers
Additional activation functions

## Compile and Run
```bash
gcc NeuralNet.c -o neural -lm
./neural