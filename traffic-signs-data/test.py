
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

# Load pickled data
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import os
# Visualizations will be shown in the notebook.


# TODO: Fill this in based on where you saved the training and testing data


testing_file = "test.p"



with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']

import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader = tf.train.import_meta_graph('asd.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))