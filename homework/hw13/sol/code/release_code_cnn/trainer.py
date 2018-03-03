import tensorflow as tf
import datetime
import os
import sys
import argparse
import numpy as np

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):

     
        self.net = net
        self.data = data
       
        self.max_iter = 3000
        self.summary_iter = 200
        


      
        self.learning_rate = 0.1
       
        self.saver = tf.train.Saver()
      
        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)
        

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    


    def optimize(self):

        self.train_losses = []
        self.test_losses = []

        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test loss through out the process
        '''
        f = open("accuracy.txt", "w")

        for i in range(self.max_iter):
            print("Iter " + str(i) + ": ", end="")
            train_batch = self.data.get_train_batch()
            train_images = np.array( [ train_batch[j]["features"] for j in range(self.data.batch_size) ] )
            train_labels = np.array( [ train_batch[j]["label"] for j in range(self.data.batch_size) ] )
            self.sess.run(self.train, feed_dict={self.net.images: train_images, self.net.labels: train_labels})
            train_accuracy = self.sess.run(self.net.accurracy, 
                feed_dict={self.net.images: train_images, self.net.labels: train_labels})
            self.train_losses.append(train_accuracy)

            val_batch = self.data.get_validation_batch()
            val_images = np.array( [ val_batch[j]["features"] for j in range(self.data.val_batch_size) ] )
            val_labels = np.array( [ val_batch[j]["label"] for j in range(self.data.val_batch_size) ] )
            prediction = self.sess.run(self.net.logits, feed_dict={self.net.images: val_images})
            val_accuracy = self.sess.run(self.net.get_acc(val_labels, prediction), 
                feed_dict={self.net.images: val_images})
            print(train_accuracy, val_accuracy)
            self.test_losses.append(val_accuracy)

            f.write(str(i) + " " + str(train_accuracy) + " " + str(val_accuracy) + "\n")

        # self.saver.save(self.sess, "my-model", global_step=5000)

        # with open("accuracy.txt", "w") as f:
        #     for i, train, val in enumerate(zip(self.train_losses, self.test_losses)):
        #         f.write(str(i) + " " + str(train) + " " + str(val) + "\n")
