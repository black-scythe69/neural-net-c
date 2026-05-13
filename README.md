# neural-net-c
# Neural Network in C from Scratch

A feedforward neural network built entirely in C with no ML libraries.
Solves the XOR problem using backpropagation and SGD.

## Architecture
- Input layer: 2 neurons
- Hidden layer: 2 neurons (sigmoid activation)
- Output layer: 1 neuron (sigmoid activation)

## Results after 10000 epochs
[0,0] -> 0.069898
[0,1] -> 0.929183
[1,0] -> 0.986103
[1,1] -> 0.011986

## Compile and Run
gcc NeuralNet.c -o neural -lm
./neural