from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.learning_algorithms import *
from nimblenet.neuralnet import NeuralNet
from nimblenet.preprocessing import construct_preprocessor, standarize
from nimblenet.data_structures import Instance
from nimblenet.tools import print_test
from read_img import read_img

# Training set
inputLayerSize, outputLayerSize, dataset = read_img.readDataSource()

preprocess          = construct_preprocessor( dataset, [standarize] )
training_data       = preprocess( dataset )
test_data           = preprocess( dataset )

cost_function = cross_entropy_cost
settings = {
    # Required settings
    "n_inputs": inputLayerSize,  # Number of network input signals
    "layers": [(3, sigmoid_function), (outputLayerSize, sigmoid_function)],
    # [ (number_of_neurons, activation_function) ]
    # The last pair in the list dictate the number of output signals

    # Optional settings
    "initial_bias_value": 0.0,
    "weights_low": -0.1,  # Lower bound on the initial weight value
    "weights_high": 0.1,  # Upper bound on the initial weight value
}

# initialize the neural network
network = NeuralNet(settings)
network.check_gradient(training_data, cost_function)

## load a stored network configuration
# network           = NeuralNet.load_network_from_file( "network0.pkl" )


# Train the network using backpropagation
RMSprop(
    network,  # the network to train
    training_data,  # specify the training set
    test_data,  # specify the test set
    cost_function,  # specify the cost function to calculate error

    ERROR_LIMIT=0.2,  # define an acceptable error limit
    # max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
    batch_size=0,  # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
    print_rate=1000,  # print error status every `print_rate` epoch.
    learning_rate=0.3,  # learning rate
    momentum_factor=0.9,  # momentum
    input_layer_dropout=0.0,  # dropout fraction of the input layer
    hidden_layer_dropout=0.0,  # dropout fraction in all hidden layers
    save_trained_network=False  # Whether to write the trained weights to disk
)



