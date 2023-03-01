import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from model_functions import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#   preprocessing data
x_train.shape = (60000, 28, 28)
x_test.shape = (10000, 28, 28)
y_train.shape = (60000, 1, 1)
y_test.shape = (10000, 1, 1)

x_train_bin = np.array(x_train > 0.5, dtype=np.float32)
x_test_bin = np.array(x_test > 0.5, dtype=np.float32)

model_dict = {}

for act_fn in all_act_fns():
    for fnl_fn in all_fnl_fns():
        for opt_fn in all_opt_fns():
                curr_config = (act_fn, fnl_fn, opt_fn)
                model_temp = create_basic_model(act_fn, fnl_fn, opt_fn, "sparse_categorical_crossentropy", verbose=False)
                model_temp.fit(x=x_train_bin, y=y_train, epochs=2, verbose=0)
                loss_temp, acc_temp = model_temp.evaluate(x=x_test_bin, y=y_test, verbose=0)
                model_dict[curr_config] = (loss_temp, acc_temp)


print(model_dict)