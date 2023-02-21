import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from model_functions import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#   preprocessing data
x_train.shape = (60000, 28, 28)
x_test.shape = (10000, 28, 28)
y_train.shape = (60000, 1, 1)
y_test = (10000, 1, 1)

x_train_bin = np.array(x_train > 0.5, dtype=np.float32)
x_test_bin = np.array(x_test > 0.5, dtype=np.float32)

#   displaying image as example
# INDEX_TO_BE_DISPLAYED = 27
# temp_image = x_train[INDEX_TO_BE_DISPLAYED]
# temp_image = np.array(temp_image, dtype="float")
# temp_image_pixels = temp_image.reshape((28, 28))
# plt.imshow(temp_image_pixels, cmap="Greys")
# plt.show()
# print(y_train[INDEX_TO_BE_DISPLAYED][0][0])

#   model testing

print("                             ------------------------------------")

for k_i_fn in all_k_i_fns():
    for act_fn in all_act_fns():
        for opt_fn in all_opt_fns():
            for loss_fn in all_loss_fns():
                model = create_basic_model(k_i_fn, act_fn, opt_fn, loss_fn)
                model.fit(x=x_train_bin, y=y_train, epochs=10, verbose=2)
                print(f"	Kernel initialiser: {k_i_fn} ; Activation function: {act_fn} ; Optimizer function: {opt_fn} ; Loss function: {loss_fn}")
                print("                             ###################################")