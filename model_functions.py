import tensorflow as tf

def create_model(activation_function: str):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, (3,3), activation=str(activation_function), input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(256, (3,3), activation=str(activation_function), input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=str(activation_function)),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
    )
    return model

def all_act_fns() -> list:
    # returns all the activation functions
    return ["linear", "sigmoid", "hard_sigmoid", "tanh", "softmax",
            "relu", "leaky_selu", "elu", "selu", "swish"]

def all_opt_fns() -> list:
    # returns all the optimizer functions
    return []

def all_loss_fns() -> list:
    # returns all the loss functions
    return []

def CNN_model():
    NUMBER_OF_OUTPUTS = 10

    MNIST_model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="sigmoid"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(NUMBER_OF_OUTPUTS, activation="softmax")
    ])

    MNIST_model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )