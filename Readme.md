## Implementation of a Fully Connected Neural Network

- The layer numbers and the neurons of each layer are setable.
- The activation fuction is choosenable, including Sigmoid, Relu and Tanh.
- Batch Gradient Descent and Mini-batch Gradient Descent Optimizer are privided, of which the iteration times, mini-batch size are setable.
- Performance and Decision boundary visulizations are provided.

Functions:

- train_model(iterations, X, y, layer_structure, learning_rate, activation_choice)
```py
    Description: train model using Batch GD optimizer
    :param iterations: iteration times.
    :param X: Training set, row vector for each data.
    :param y: Label, in one-hot form.
    :param layer_structure: Network Parms, including layers, numbers of Neuron of each layer.
    :param activation_choice: Activation function of hidden layers.
    :return: The trained model param: Weights, biases; costs.
```

- train_model_minibatch(epochs, mini_batch_size, X, y, layer_structure, learning_rate, activation_choice)
```py
    Description: train model using Mini-Batch GD optimizer 
    :param epochs: iteration times of the whole training set.
    :param mini_batch_size: as the variable name.
    :param X: Training set, row vector for each data.
    :param y: Label, in one-hot form.
    :param layerstructure: Network Parms, including layers, numbers of Neuron of each layer.
    :param activation_choice: Activation_choice: activation function of hidden layers.
    :return: The trained model param: Weights, biases; costs.
```