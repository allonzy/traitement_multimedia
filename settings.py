classes_to_predict = ["cat","flower"]
from nimblenet.activation_functions import sigmoid_function

nimbles_settings = {
    # Required settings
    "layers": [(3, sigmoid_function), (1, sigmoid_function)],
    # [ (number_of_neurons, activation_function) ]
    # The last pair in the list dictate the number of output signals

    # Optional settings
    "initial_bias_value": 0.0,
    "weights_low": -0.1,  # Lower bound on the initial weight value
    "weights_high": 0.1,  # Upper bound on the initial weight value
}
imgread_settings = {
    "size": (32,32),
    "black_and_white": False,
    "data_source": "res/*",
    "img_ext": [".png",".jpg"],
}