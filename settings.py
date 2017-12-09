from nimblenet.activation_functions import sigmoid_function

classes_to_predict = ["dandelion","roses","sunflowers","tulips","daisy"]

imgread_settings = {
    "size": (32,32),
    "black_and_white": True,
    "data_source": "res/*",
    "img_ext": [".png",".jpg"],
}