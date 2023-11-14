## Hardware Architectures of Deep Learning

## Lab1:
**Content of each of the Python Notebooks**
1. Part1_Parametrized_CNN : This notebooks computes the accuracy of several CNNs focusing on its development when the number of epochs used for training is varied.
2. Part1_AccuracyGraphs_ConstantEpochs: This notebook computes the bar graphs showed in the report. The plots showcase the variation of accuracy based on the variation of different parameters for a constant number of epochs.
3.test
4.
5.

**Description for Part 1 code**

Parameterized MLP and CNN module used to test the performance on small dataset CIFAR10 and MNIST. Implemented with PyTorch.

```python
# Multilayer perceptron model
MLP(n_hidden_layers, hidden_neurons, input_size, n_classes)
```

```python
# Convolutional neural network model
ConvNet(n_conv, n_fc, conv_ch, filter_size, fc_size, pooling_size, input_size, input_channels, n_classes, activation_fn) #same usage as LeNet
```

```python
# Training function
train(model_params, model_name, device, epochs)
```

**MLP**
|Parameter|Description|
|---|----|
|n_hidden_layers|number of hidden layers|
|hidden_neurons|list of size of hidden layers|
|input_size|dimension of input data|
|n_classes|categories of output classes|

**ConvNet**
|Parameter|Description|
|----|---|
|n_conv|number of convolutional layers|
|n_fc|number of full connected layers|
|conv_ch|list of channels of each conventional layer|
|filter_size|list of filter size of each conventional layer|
|fc_size|list of size of each full connected layer|
|pooling_size|shared size of squared pooling layer|
|input_size|dimension of input data|
|input_channels|channels of input data|
|n_classes|categories of output classes|
|activation_fn|type of activation function|

**train**
|Parameter|Description|
|---|----|
|model_params|dict of parameters unwrapped when instantiated a new model|
|model_name|name of model, used to specific the path to load or store the model|
|device|designate the device to be trained on|
|epochs|total training epochs|
