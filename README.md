# MNIST-CNN
Convolutional Neural Network written using the PyTorch API designed to recognize data from the MNIST dataset.

## Purpose
This model's purpose is to classify images from the MNIST dataset with great accuracy.

## Model

### Convolutional and Pooling Layers
This layer takes in an MNIST image in the form of a tensor of size (batch_size, 1 , 28, 28) and applies a convolutional layer where kernel size = 5 and stride = 2. ReLU is used as the activation function. Then, a pooling layer is applied where kernel size = 2 and stride = 2. This sequence is repeated a second time.

### Fully Connected Layers
4 fully connected layers are used.

Layer 1 size: 384  
Layer 2 Size: 1124  
Layer 3 Size: 2248  
Layer 4 Size: 10  

Layers 1 through 3 use ReLU as an activation function.
Using LogSoftmax for the output layer is an option. To do this, simply change the forward() function's return value from `return x` to `return LogSoftmax(x)`.

## Training

### Loss Function and Optimizer
This model uses Cross Entropy Loss as a loss function and Stochastic Gradient Descent as an optimizer.

### Training Parameters
```python
train_batch_size = 32
learning_rate = 0.01
momentum = 0.8
num_epochs = 200
```
