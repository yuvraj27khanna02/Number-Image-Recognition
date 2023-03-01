import tensorflow as tf
from scipy.ndimage import shift
import numpy as np
from matplotlib import pyplot as plt

def create_basic_model(act_fn, fnl_fn, opt_fn, loss_fn, verbose=False):
    """Returns a basic CNN model
    """
    if verbose:
        print(f"            Activation function: {act_fn} ; Optimizer function: {opt_fn} ; Loss function: {loss_fn}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation=act_fn, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation=act_fn),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation=act_fn),
        
        tf.keras.layers.Dense(10, activation=fnl_fn)
    ])
    model.compile(optimizer=opt_fn, loss=loss_fn, metrics=["accuracy"])
    return model

def create_BN_model(k_i_fn, act_fn, opt_fn, loss_fn, verbose=False):
    """Returns a basic CNN model with Batch Normalization
    """
    if verbose:
        print(f"	Kernel initialiser: {k_i_fn} ; Activation function: {act_fn} ; Optimizer function: {opt_fn} ; Loss function: {loss_fn}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation=act_fn, input_shape=(28, 28, 1), kernel_initializer=k_i_fn),
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

# def all_adv_act_fns() -> list:
#     """Returns all Advanced activation functions
#     """
#     return [tf.keras.layers.ReLU(), tf.keras.layers.Softmax(),
#     		tf.keras.layers.LeakyReLU(), tf.keras.layers.PReLU(),
#             tf.keras.layers.ELU(), tf.keras.layers.ThresholdedReLU()]

def all_act_fns() -> list:
    """Returns all tf.keras activation functions
    """
    return ["elu", "exponential", "gelu", "hard_sigmoid", "linear",
            "relu", "selu", "sigmoid", "softmax", "softplus", "softsign",
            "swish", "tanh"]

def all_fnl_fns() -> list:
    """Returns all tf.keras activation fucntions for the last layer of a neural network
    """
    return ["sigmoid", "softmax", "softplus", "softsign"]

def all_opt_fns() -> list:
    # returns all the optimizer functions
    return ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'deserialize', 'get', 'serialize']

# def all_loss_fns() -> list:
#     """Returns all loss functions
#     >>> all_loss_fns = ['binary_crossentropy', 'binary_focal_crossentropy', 'categorical_crossentropy', 'hinge', 'kl_divergence', 
#     'kld', 'kullback_leibler_divergence', 'log_cosh', 'logcosh', 'mae', 'mape', 'mean_absolute_error', 'mean_absolute_percentage_error',
#     'mean_squared_error', 'mean_squared_logarithmic_error', 'mse', 'msle', 'poisson', 'sparse_categorical_accuracy', 
#     'sparse_categorical_crossentropy', 'sparse_top_k_categorical_accuracy', 'squared_hinge', 'top_k_categorical_accuracy']
#     """
#     # returns all the loss functions
#     return ['sparse_categorical_crossentropy',]

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

def display_image(temp_image):
    temp_image = np.array(temp_image, dtype="float")
    temp_image_pixels = temp_image.reshape((28, 28))
    plt.imshow(temp_image_pixels, cmap="Greys")
    plt.show()