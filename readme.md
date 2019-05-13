## Implementation of a Fully Connected Neural Network.

- The layer numbers and the neurons of each layerare setable.
- The activation fuction is choosenable, including Sigmoid, Relu and Tanh.
- Batch Gradient Descent and Mini-batch Gradient Descent Optimizer are privided, of which the iteration times, mini-batch size are setable.
- Performance and Decision boundary visulizations are provided.

functions:

- train_model(iterations, X, y, layer_structure, learning_rate, activation_choice)
"""
    Description: train model using Batch GD optimizer
    :param iterations: iteration times
    :param layer_structure: Network Parms, including layers, numbers of Neuron of each layer
    :param X: Training set, row vector for each data
    :param y: label, one-hot form
    :param activation_choice: activation function of hidden layers
    :return: the trained model param: Weights, biases; costs
"""

- train_model_minibatch(epochs, mini_batch_size, X, y, layer_structure, learning_rate, activation_choice)
"""
    Description: train model using Mini-Batch GD optimizer 
    :param epochs: iteration times of the whole training set
    :param mini_batch_size: as the name
    :param X: Training set, row vector for each data
    :param y: label, one-hot form
    :param layerstructure: Network Parms, including layers, numbers of Neuron of each layer
    :param activation_choice: activation_choice: activation function of hidden layers
    :return: the trained model param: Weights, biases; costs
"""