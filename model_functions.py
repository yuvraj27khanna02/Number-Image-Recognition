#import tensorflow as tf

def create_basic_model(k_i_fn, act_fn, opt_fn, loss_fn):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, (3,3), activation=act_fn,
        	input_shape=(28, 28, 1), kernel_initializer=k_i_fn),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(256, (3,3), activation=act_fn, 
        	input_shape=(28, 28, 1), kernel_initializer=k_i_fn),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=act_fn),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
    optimizer=opt_fn,
    loss=loss_fn,
    metrics=["accuracy"]
    )
    return model

def all_k_i_fns () -> list:
    """Returns a list of all kernel initialiser functions
    """
    # for all k_i in all_k_i_fns():
    #    tf.keras.layer.(kernel_initializer=k_i)
    #    mode.add(layers.Dense(kernel_initializer=k_i))
    return [tf.keras.initializers.RandomNormal(), 
    		tf.keras.initializers.RandomUniform(),
            tf.keras.initializers.TruncatedNormal(),
            tf.keras.initializers.Zeros(),
            tf.keras.initializers.Ones(),
            tf.keras.initializers.GlorotNormal()
            tf.keras.initializers.GlorotUniform(),
            tf.keras.initializers.HeNormal(),
            tf.keras.initializers.HeUniform(),
            tf.keras.initializers.Identity(),
            tf.keras.initializers.Orthogonal(),
            tf.keras.initializers.Constant(),
            tf.keras.initializers.VaruanceScaling()
            tf.keras.initializers.Uniform()]

def all_adv_act_fns() -> list:
    """Returns all Advanced activation functions
    """
    return [tf.keras.layers.ReLU(), tf.keras.layers.Softmax(),
    		tf.keras.layers.LeakyReLU(), tf.keras.layers.PReLU(),
            tf.keras.layers.ELU(), tf.keras.layers.ThresholdedReLU()]

def all_act_fns() -> list:
    """Returns all tf.keras activation functions
    """
    return [tf.keras.activations.deserialize(),
    		tf.keras.activations.elu(),
            tf.keras.activations.exponential(),
            tf.keras.activations.gelu(),
            tf.keras.activations.hard_sigmoid(),
            tf.keras.activations.linear(),
            tf.keras.activations.relu(),
            tf.keras.activations.selu(),
            tf.keras.activations.serialize(),
            tf.keras.activations.sigmoid(),
            tf.keras.activations.softmax(),
            tf.keras.activations.softplus(),
            tf.keras.activations.softsign(),
            tf.keras.activations.swish(),
            tf.keras.activations.tanh()]

def all_opt_fns() -> list:
    # returns all the optimizer functions
    return [opt, tf.keras.optmiziers.Adam(),
    		tf.keras.optimizers.SGD(),
            tf.keras.optimizers.RMSprop(),
            tf.keras.optimizers.AdamW(),
            tf.keras.optimizers.Adadelta(),
            tf.keras.optimizers.Adagrad(),
            tf.keras.optimizers.Adamax(),
            tf.keras.optimizers.Adafactor(),
            tf.keras.optimizers.Nadam(),
            tf.keras.optimizers.Ftrl()]

def all_loss_fns() -> list:
    # returns all the loss functions
    return [tf.keras.metrics.kl_divergence(),
    		tf.keras.metrics.mean_absolute_error(),
            tf.keras.metrics.mean_sqaured_error(),
            tf.keras.metrics.mean_absolute_percentage_error(),
            tf.keras.metrics.mean_sqaured_logarithmic_error(),
            tf.keras.metrics.binary_crossentropy(),
            tf.keras.metrics.binary_focal_crossentropy(),
            tf.keras.metrics.categorical_crossentropy(),
            tf.keras.metrics.categorical_hinge(),
            tf.keras.metrics.cosine_similarity(),
            tf.keras.metrics.deserialize(),
            tf.keras.metrics.hinge(),
            tf.keras.metrics.huber(),
            tf.keras.metrics.kl_divergence(),
            tf.keras.metrics.kld(),
            tf.keras.metrics.kullback_leibler_divergence(),
            tf.keras.metrics.log_cosh(),
            tf.keras.metrics.poisson(),
            tf.keras.metrics.serialize(),
            tf.keras.metrics.sparse_categorical_crossentropy(),
            tf.keras.metrics.sqaured_hinge()]

def all_pds() -> list:
    """Returns all padding options
    """
    return ["valid", "same"]

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