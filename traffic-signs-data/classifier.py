
# Load pickled data
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import os
# Visualizations will be shown in the notebook.


# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"   ## I used my extended datase ... script given later in the notebook.. plz use it to test 
testing_file = "test.p"
validation_file = "valid.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

import cv2
def RGB2YCrCb444(img):
   return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
from sklearn.model_selection import train_test_split

#X_train,X_validation,y_train,y_validation=train_test_split(X_train, y_train, test_size=0.2, random_state=42)
with open(validation_file, mode='rb') as f:
    valida = pickle.load(f)
X_validation,y_validation =  valida['features'], valida['labels']

from tensorflow.contrib.layers import flatten

import tensorflow as tf

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.03

    convaz0_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 3, 6), mean = mu, stddev = sigma))
    convaz0_b = tf.Variable(tf.zeros(6))
    convaz0   = tf.nn.conv2d(x, convaz0_W, strides=[1, 1, 1, 1], padding='SAME') + convaz0_b
    convaz0 = tf.nn.relu(convaz0)

    convaaz0_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 3, 6), mean = mu, stddev = sigma))
    convaaz0_b = tf.Variable(tf.zeros(6))
    convaaz0   = tf.nn.conv2d(x, convaaz0_W, strides=[1, 1, 1, 1], padding='SAME') + convaaz0_b
    convaaz0 = tf.nn.relu(convaaz0)

    conva0_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 6), mean = mu, stddev = sigma))
    conva0_b = tf.Variable(tf.zeros(6))
    conva0   = tf.nn.conv2d(convaaz0, conva0_W, strides=[1, 1, 1, 1], padding='SAME') + conva0_b
    conva0 = tf.nn.relu(conva0)

    convca0_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 6), mean = mu, stddev = sigma))
    convca0_b = tf.Variable(tf.zeros(6))
    convca0   = tf.nn.conv2d(convaaz0, convca0_W, strides=[1, 1, 1, 1], padding='SAME') + convca0_b
    convca0 = tf.nn.relu(convca0)

    conv0_W = tf.Variable(tf.truncated_normal(shape=(7, 7, 6, 6), mean = mu, stddev = sigma))
    conv0_b = tf.Variable(tf.zeros(6))
    conv0   = tf.nn.conv2d(convaaz0, conv0_W, strides=[1, 1, 1, 1], padding='SAME') + conv0_b
    conv0 = tf.nn.relu(conv0)
    
    concatenated_tensor = tf.concat(3,[conv0, conva0,convaz0,convca0])

    conv1da_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 24, 32), mean = mu, stddev = sigma))
    conv1da_b = tf.Variable(tf.zeros(32))
    conv1da   = tf.nn.conv2d(concatenated_tensor, conv1da_W, strides=[1, 1, 1, 1], padding='SAME') + conv1da_b
    conv1da = tf.nn.relu(conv1da)
    conv1da = tf.nn.max_pool(conv1da, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')



    conv1a_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 24, 32), mean = mu, stddev = sigma))
    conv1a_b = tf.Variable(tf.zeros(32))
    conv1a   = tf.nn.conv2d(concatenated_tensor, conv1a_W, strides=[1, 1, 1, 1], padding='SAME') + conv1a_b
    conv1a = tf.nn.relu(conv1a)
    conv1a = tf.nn.max_pool(conv1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(concatenated_tensor, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    conv1aaa=tf.concat(3,[conv1a, conv1,conv1da])

    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 96, 112), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(112))
    conv2   = tf.nn.conv2d(conv1aaa, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    fca0   = flatten(conv2)

    fc0_W = tf.Variable(tf.truncated_normal(shape=(5488, 1152), mean = mu, stddev = sigma))
    fc0_b = tf.Variable(tf.zeros(1152))
    fc0   = tf.matmul(fca0, fc0_W) + fc0_b
    fc0    = tf.nn.relu(fc0)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1152, 600), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(600))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(600, 130), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(130))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(130, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

### Train your model here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

EPOCHS = 12
BATCH_SIZE = 64
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
rate = 0.00045
beta = 0.01

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy_operation_out = [tf.reduce_mean(tf.cast(correct_prediction, tf.float32)),cross_entropy]

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
	
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    maxa=0.000
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
       # for offset in range(0, num_examples, BATCH_SIZE):
        for offset in range(0, BATCH_SIZE, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            for dd in range(len(batch_x)):
                batch_x[i]=RGB2YCrCb444(batch_x[i])
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print()

        if(test_accuracy>maxa):
           maxa=test_accuracy
           a=tf.train.Saver()
           save_path='C:\research\YUVprecision\traffic-signs-data\bbb.ckpt'
           if not os.path.isabs(save_path):
                save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))
           a.save(sess,save_path)
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    save_path='C:\research\YUVprecision\traffic-signs-data\lenet2'
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))	
    saver.save(sess, 'lenet2')
    print("Model saved")