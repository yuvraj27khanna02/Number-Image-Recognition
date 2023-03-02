import tensorflow as tf
from scipy.ndimage import shift
import numpy as np
from matplotlib import pyplot as plt
import heapq

def create_basic_model(act_fn, fnl_fn, opt_fn, loss_fn, input_shape, verbose=False):
    """Returns a basic CNN model.
    """
    if verbose:
        print(f"            Activation function: {act_fn} ; Final Activation function: {fnl_fn} ; Optimizer function: {opt_fn} ; Loss function: {loss_fn}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation=act_fn, input_shape=input_shape),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation=act_fn),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation=act_fn),
        
        tf.keras.layers.Dense(10, activation=fnl_fn)
    ])
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=["accuracy"])
    return model

def create_BN_model(k_i_fn, act_fn, opt_fn, loss_fn, input_shape, verbose=False):
    """Returns a basic CNN model with Batch Normalization.
    Better than create_basic_model but requires more processing as well.
    """
    if verbose:
        print(f"	Kernel initialiser: {k_i_fn} ; Activation function: {act_fn} ; Optimizer function: {opt_fn} ; Loss function: {loss_fn}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation=act_fn, input_shape=input_shape, kernel_initializer=k_i_fn),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation=act_fn),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation=act_fn),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=["accuracy"])
    return model

def all_k_i_fns() -> list:
    """Returns a list of all kernel initialiser functions
    >>> for k_i in all_k_i_fns():
    >>>     tf.keras.layers.Conv2D(kernel_initializer=k_1)
    """
    return ['constant', 'deserialize', 'get', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
            'identity', 'lecun_normal', 'lecun_uniform', 'ones', 'orthogonal', 'random_normal', 'random_uniform',
            'serialize', 'truncated_normal', 'variance_scaling', 'zeros']

def all_act_fns() -> list:
    """Returns all tf.keras activation functions
    """
    return ["elu", "exponential", "gelu", "hard_sigmoid", "linear",
            "relu", "selu", "sigmoid", "softmax", "softplus", "softsign",
            "swish", "tanh"]

def all_fnl_fns() -> list:
    """Returns all tf.keras activation fucntions for the last layer of a neural network
    """
    return ["sigmoid", "softmax", "softplus", "softsign", "elu"]

def all_opt_fns() -> list:
    """Returns all arguements for optimizer parameter of tf.keras.Model.compile
    full_list -> ["adam", "adamax", "adagrad", "adadelta", "nadam", "ftrl", "rsmprop", "sgd"]
    """
    # returns all the optimizer functions
    return ["adamax", "adagrad", "adadelta", "adam", "ftrl", "nadam", "rmsprop", "sgd"]


def all_loss_fns() -> list:
    """Returns all loss functions
    """
    return ['binary_crossentropy', 'binary_focal_crossentropy', 'categorical_crossentropy', 'hinge', 'kl_divergence',
            'kld', 'kullback_leibler_divergence', 'log_cosh', 'logcosh', 'mae', 'mape', 'mean_absolute_error', 'mean_absolute_percentage_error',
            'mean_squared_error', 'mean_squared_logarithmic_error', 'mse', 'msle', 'poisson', 'sparse_categorical_accuracy', 
            'sparse_categorical_crossentropy', 'sparse_top_k_categorical_accuracy', 'squared_hinge', 'top_k_categorical_accuracy']

def all_pds() -> list:
    """Returns all padding options
    """
    return ["valid", "same"]
    
# Functions for Data Augmentation
    
def _shift_image(image, diffx, diffy, image_size):
    curr_image = image.reshape(image_size)
    shifted_image = shift(curr_image, [diffx, diffy], cval=0, mode="constant")
    return shifted_image.reshape([-1])
    
def augment_data(x_train, y_train):
    x_train_aug = [image for image in x_train]
    y_train_aug = [image for image in y_train]
    for diffx, diffy in ((1,0), (-1,0), (0,1), (0,-1)):
        for (x_image, y_label) in (x_train, y_train):
            x_train_aug.append(_shift_image(x_image, diffx, diffy))
            y_train_aug.append(_shift_image(y_label))

def display_image(temp_image, input_shape):
    temp_image = np.array(temp_image, dtype="float")
    temp_image_pixels = temp_image.reshape(input_shape)
    plt.imshow(temp_image_pixels, cmap="Greys")
    plt.show()

def get_max_n_keys(model_dict, n) -> list:
    """Returns max n keys for comparing model accuracy.
    n < len(model_dict)
    """
    max_5_vals = heapq.nlargest(n, [val[1] for val in model_dict.values()])
    return_list = []
    for k, v in model_dict.items():
        if v[1] in max_5_vals:
            return_list.append(k)
    return return_list

def get_min_n_keys(model_dict, n) -> list:
    """Returns min n keys for comparing model loss.
    n < len(model_dict)
    """
    min_5_vals = heapq.nsmallest(n, [val[0] for val in model_dict.values()])
    return_list = []
    for k, v in model_dict.items():
        if v[0] in min_5_vals:
            return_list.append(k)
    return return_list


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from model_functions import *
import heapq
from typing import Tuple

def CNN_all_parameters_(x_train, x_test, y_train, y_test, input_shape, verbose=False) -> Tuple:
    """Returns the best parameters of a CNN model using user given dataset
    """
    model_res_dict = {}
    for opt_fn in all_opt_fns():
        for fnl_fn in all_fnl_fns():
            for act_fn in all_act_fns():
                    curr_config = (act_fn, fnl_fn, opt_fn)
                    model_temp = create_basic_model(act_fn, fnl_fn, opt_fn, "sparse_categorical_crossentropy", input_shape, verbose=verbose)
                    model_temp.fit(x=x_train, y=y_train, epochs=2, verbose=int(verbose))
                    loss_temp, acc_temp = model_temp.evaluate(x=x_test, y=y_test, verbose=int(verbose))
                    model_res_dict[curr_config] = (loss_temp, acc_temp)
    best_accuracy_config = get_max_n_keys(model_res_dict, 5)
    best_loss_config = get_min_n_keys(model_res_dict, 5)
    return best_accuracy_config, best_loss_config