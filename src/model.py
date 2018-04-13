import os
import tensorflow as tf
import numpy as np
import argparse
import math, time
import scipy
from utility import BatchGenerator
from operations import *

CHANNELS = 4

# WHAT DO WE NEED?
# GENERATOR
# DISCRIMINATOR
class CGAN:
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.save_folder = args.save_folder
        self.reload = args.reload
        self.labelSize = labelSize
        self.sess = sess
        self.buildModel()

        return

    def _fc_variable(self, weight_shape, name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def calcImageSize(self,dh,dw,stride):
        return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildGenerator(self, z, label, reuse=False, isTraining=True):
        dim_0_h, dim_0_w = self.imageSize[0], self.imageSize[1] # Building backwards convolutional layers
        dim_1_h, dim_1_w = self.calcImageSize(dim_0_h, dim_0_w, stride=2)
        dim_2_h, dim_2_w = self.calcImageSize(dim_1_h, dim_1_w, stride=2)
        dim_3_h, dim_3_w = self.calcImageSize(dim_2_h, dim_2_w, stride=2)
        dim_4_h, dim_4_w = self.calcImageSize(dim_3_h, dim_3_w, stride=2)

        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()

            yb = tf.reshape(label, [self.batch_size, 1, 1, self.labelSize])
            # l = tf.one_hot(label, self.labelSize, name="label_onehot")
            h = tf.concat([z, label], axis=1, name="concat_z")

            # fc1
            self.g_fc1_w, self.g_fc1_b = self._fc_variable([self.zdim + self.labelSize, 512 * dim_4_h * dim_4_w],name="fc1")
            h0 = tf.matmul(h, self.g_fc1_w) + self.g_fc1_b
            h0 = tf.nn.relu(h0)
            #
            h0 = tf.reshape(h0, (self.batch_size, dim_4_w, dim_4_h, 512))
            h0 = conv_cond_concat(h0, yb)

            h0 = deconv2d(h0, output_shape=[self.batch_size, dim_3_h, dim_3_w, 256], name="deconv4")
            h0 = batch_norm(h0, is_training=isTraining, scope="gNorm4")
            # h0 = tf.nn.dropout(h0, 0.5)
            h0 = tf.nn.relu(h0)
            h0 = conv_cond_concat(h0, yb)

            # Deconv 1 -> 128x128
            h1 = deconv2d(h0, output_shape=[self.batch_size, dim_2_h, dim_2_w, 128], name="deconv3")
            h1 = batch_norm(h1, is_training=isTraining, scope="gNorm3")
            # h1 = tf.nn.dropout(h1, 0.5)
            h1 = tf.nn.relu(h1)
            h1 = conv_cond_concat(h1, yb)

            # Deconv 2 -> 64x64
            h2 = deconv2d(h1, output_shape = [self.batch_size, dim_1_h, dim_1_w, 64], name="deconv2")
            h2 = batch_norm(h2, is_training=isTraining, scope="gNorm2")
            # h2 = tf.nn.dropout(h2, 0.5)
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            # Deconv 3 -> 64x64 Image
            h3 = deconv2d(h2, output_shape=[self.batch_size, dim_0_h, dim_0_w, CHANNELS], name= "deconv1")

            # Using tanh as per improved DCGAN paper.
            y = tf.nn.sigmoid(h3)

        return y

    def buildDiscriminator(self, y, label, reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse: scope.reuse_variables()

            # conditional layer
            yb = tf.reshape(label, [self.batch_size, 1, 1, self.labelSize])
            x = conv_cond_concat(y, yb)

            # conv1 CHANNELS + self.labelSize
            h0 = conv2d(x, output_shape=[5, 5, 4 + self.labelSize, 64], name="conv1")
            h0 = batch_norm(h0, is_training=self.isTraining, scope="dNorm1")
            h0 = lrelu(h0)
            h0 = conv_cond_concat(h0, yb)

            # conv2
            h1 = conv2d(h0, output_shape=[5, 5, 64 + self.labelSize, 128], name="conv2")
            h1 = batch_norm(h1, is_training=self.isTraining, scope="dNorm2")
            h1 = lrelu(h1)
            h1 = conv_cond_concat(h1, yb)

            # conv3
            h2 = conv2d(h1, output_shape=[5, 5, 128 + self.labelSize, 256], name="conv3")
            h2 = batch_norm(h2, is_training=self.isTraining, scope="dNorm3")
            h2 = lrelu(h2)
            h2 = conv_cond_concat(h2, yb)

            # conv4
            h3 = conv2d(h2, output_shape=[5, 5, 256 + self.labelSize, 512], name="conv4")
            h3 = batch_norm(h3, is_training=self.isTraining, scope="dNorm4")
            h3 = lrelu(h3)
            h3 = conv_cond_concat(h3, yb)

            # fc1
            n_b, n_h, n_w, n_f = [int(x) for x in h3.get_shape()]
            h3 = tf.reshape(h3,[self.batch_size,n_h*n_w*n_f])
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([n_h*n_w*n_f,1],name="fc1")
            h3 = tf.matmul(h3, self.d_fc1_w) + self.d_fc1_b

        return h3

    def buildModel(self):
        # define variables
        self.z      = tf.placeholder(tf.float32, [self.batch_size, self.zdim], name="z")
        self.l      = tf.placeholder(tf.float32  , [self.batch_size, self.labelSize], name="label")

        img_dimensions = [self.imageSize[0], self.imageSize[1], CHANNELS]
        self.g_real = tf.placeholder(tf.float32, [self.batch_size] + img_dimensions, name="images")

        self.g_fake = self.buildGenerator(self.z, self.l)
        self.g_sample = self.buildGenerator(self.z, self.l, reuse=True, isTraining=False)

        self.d_real  = self.buildDiscriminator(self.g_real, self.l)
        self.d_fake  = self.buildDiscriminator(self.g_fake, self.l, reuse=True)

        print("BUILT MODELS")

        # define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like (self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
        self.g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like (self.d_fake)))
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        print("DEFINED LOSS FUNCTIONS")

        # define optimizer
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.5).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.5).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        print("DEFINED OPTIMIZERS")

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.save_folder: self.writer = tf.summary.FileWriter(self.save_folder, self.sess.graph)

        return

    def train(self,f_batch):
        def tileImage(imgs, size):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h, w = imgs.shape[1], imgs.shape[2]
            if (imgs.shape[3] in (3,4,1)):
                colour = imgs.shape[3]
                img = np.zeros((h * d, w * d, colour))
                for idx, image in enumerate(imgs):
                    i = idx // d
                    j = idx-i*d
                    img[j * h:j * h + h, i * w:i * w + w, :] = image
                return img
            else:
                raise ValueError('in merge(images,size) images parameter '
                                'must have dimensions: HxW or HxWx3 or HxWx4')

        if self.save_folder and not os.path.exists(os.path.join(self.save_folder,"images")):
            os.makedirs(os.path.join(self.save_folder,"images"))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.loadModel(self.reload)

        step = -1
        start = time.time()
        while True:
            step += 1

            batch_images, batch_labels = f_batch(self.batch_size)
            # Random noise vector
            batch_z                   = np.random.uniform(-1., +1., [self.batch_size, self.zdim])

            # Update generator
            _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
            _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
            feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images}
            _, d_loss, g_fake, g_real = self.sess.run([self.d_optimizer, self.d_loss, self.g_fake, self.g_real], feed_dict = feed_dict)

            if step % 100 == 0:
                # Run models outputting images as training is run.
                print ("{}: loss(D)={:.4f}, loss(G)={:.4f}; time/step = {:.2f} sec".format(step, d_loss, g_loss, time.time() - start))
                start = time.time()

                l0 = np.array([[x%2, x%7, x%7, x%13, x%13, x%13, x%13] for x in range(self.batch_size)])
                l1 = np.array([[0, x%9, x%12, x%20, x%20, x%20, x%30] for x in range(self.batch_size)])
                z1 = np.random.uniform(-1, +1, [self.batch_size, self.zdim])
                z2 = np.random.uniform(-1, +1, [self.zdim])
                z2 = np.expand_dims(z2, axis=0)
                z2 = np.repeat(z2, repeats=self.batch_size, axis=0)

                g_image1 = self.sess.run(self.g_sample, feed_dict={self.z:z1, self.l:l0})
                g_image2 = self.sess.run(self.g_sample, feed_dict={self.z:z2, self.l:l1})
                scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}d_real.png".format(step)), tileImage(g_real, [64, 64]) * 255.)
                scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}d_fake1.png".format(step)), tileImage(g_image1, [64, 64]) * 255.)
                scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}d_fake2.png".format(step)), tileImage(g_image2, [64, 64]) * 255.)
                self.saver.save(self.sess,os.path.join(self.save_folder,"model.ckpt"),step)