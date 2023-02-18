import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

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

from model_functions import create_model, all_act_fns

print("                             ------------------------------------")

for act_fn in all_act_fns():
    print(f"               <-- <-- <-- {act_fn} --> --> -->    ")
    model = create_model(str(act_fn))
    model.fit(x=x_train_bin, y=y_train, 
              epochs=1, verbose=2)
    print("                             ###################################")