import os
import tensorflow as tf
import numpy as np
import argparse
import math, time
import scipy
from utility import BatchGenerator
from operations import *

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
        self.cdim = args.cdim
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
        return int(math.ceil(float(dh)/float(stride))), int(math.ceil(float(dw)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildGenerator(self, z, y, reuse=False, isTraining=True):
        gf_dim = 64
        dim_0_h, dim_0_w = self.imageSize[0], self.imageSize[1] # Building backwards convolutional layers
        dim_1_h, dim_1_w = self.calcImageSize(dim_0_h, dim_0_w, stride=2)
        dim_2_h, dim_2_w = self.calcImageSize(dim_1_h, dim_1_w, stride=2)
        dim_3_h, dim_3_w = self.calcImageSize(dim_2_h, dim_2_w, stride=2)
        dim_4_h, dim_4_w = self.calcImageSize(dim_3_h, dim_3_w, stride=2)
        dim_5_h, dim_5_w = self.calcImageSize(dim_4_h, dim_4_w, stride=2)

        with tf.variable_scope("Generator") as scope:
            if reuse: scope.reuse_variables()

            cgan = False
            if cgan:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.labelSize]) # I.e. reshapes to [64, 1, 1, 74]
                # l = tf.one_hot(label, self.labelSize, name="label_onehot")
                z = tf.concat([z, y], axis=1, name="concat_z")

                # Two Fully Connected Layers
                h0 = fc(z, gf_dim, 'g_h0_lin')
                # h0 = batch_norm(h0, is_training=isTraining, scope="gNorm1")
                h0 = tf.nn.relu(h0)
                h0 = tf.concat([h0, y], axis=1, name="concat_a")

                h0 = fc(z, gf_dim * 16 * dim_5_h * dim_5_w, 'g_h1_line')
                # h0 = batch_norm(h0, is_training=isTraining, scope="gNorm7")
                h0 = tf.nn.relu(h0)
                h0 = tf.reshape(h0, [-1, dim_5_h, dim_5_w, gf_dim * 16])

                h0 = conv_cond_concat(h0, yb)

                h0 = deconv2d(h0, output_shape=[self.batch_size, dim_4_h, dim_4_w, gf_dim * 8], name="deconv5")
                h0 = batch_norm(h0, is_training=isTraining, scope="gNorm6")
                h0 = tf.nn.relu(h0)
                h0 = conv_cond_concat(h0, yb)

                # Deconv 1
                h0 = deconv2d(h0, output_shape=[self.batch_size, dim_3_h, dim_3_w, gf_dim * 4], name="deconv4")
                h0 = batch_norm(h0, is_training=isTraining, scope="gNorm4")
                h0 = tf.nn.relu(h0)
                h0 = conv_cond_concat(h0, yb)

                # Deconv 1
                h1 = deconv2d(h0, output_shape=[self.batch_size, dim_2_h, dim_2_w, gf_dim * 2], name="deconv3")
                h1 = batch_norm(h1, is_training=isTraining, scope="gNorm3")
                h1 = tf.nn.relu(h1)
                h1 = conv_cond_concat(h1, yb)

                # Deconv 2
                h2 = deconv2d(h1, output_shape = [self.batch_size, dim_1_h, dim_1_w, gf_dim * 1], name="deconv2", filt=[8, 8])
                h2 = batch_norm(h2, is_training=isTraining, scope="gNorm2")
                h2 = tf.nn.relu(h2)
                h2 = conv_cond_concat(h2, yb)

                # Deconv 3 -> 64x64 Image
                h3 = deconv2d(h2, output_shape=[self.batch_size, dim_0_h, dim_0_w, self.cdim], name= "deconv1")

                # Using tanh as per improved DCGAN paper.
                y = tf.nn.tanh(h3, name="sprite")
            else:
                s_h, s_w = self.imageSize[0], self.imageSize[1]
                s_h2, s_h4, s_h8 = int(s_h/2), int(s_h/4), int(s_h/8)
                s_w2, s_w4, s_w8 = int(s_w/2), int(s_w/4), int(s_w/8)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.labelSize])
                z = tf.concat([z, y], axis=1, name="concat_z")

                h0 = fc(z, 1024, 'g_h0_lin')
                h0 = batch_norm(h0, is_training=isTraining, scope="gNorm1")
                h0 = tf.nn.relu(h0)
                h0 = tf.concat([h0, y], axis=1, name="concat_h1")

                h1 = fc(h0, 64 * 2 * s_h8 * s_w8, 'g_h1_lin')
                h1 = batch_norm(h1, is_training=isTraining, scope="gNorm2")
                h1 = tf.nn.relu(h1)
                h1 = tf.reshape(h1, [self.batch_size, s_h8, s_w8, 64 * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = deconv2d(h1,[self.batch_size, s_h4, s_w4, 64 * 2], name='g_h1')
                h2 = batch_norm(h2, is_training=isTraining, scope="gNorm3")
                h2 = tf.nn.relu(h2)
                h2 = conv_cond_concat(h2, yb)

                h2 = deconv2d(h2,[self.batch_size, s_h2, s_w2, 64 * 2], name='g_h2')
                h2 = batch_norm(h2, is_training=isTraining, scope="gNorm4")
                h2 = tf.nn.relu(h2)
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.tanh(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.cdim], name='g_h3'))

        return y

    def buildDiscriminator(self, z, y, reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse: scope.reuse_variables()
            df_dim = 64

            cgan_layer = True
            if cgan_layer:
                # This is based on the CGAN architecture

                # conditional layer
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.labelSize])
                z = conv_cond_concat(z, yb)

                # conv1 CHANNELS + self.labelSize
                h0 = conv2d(z, output_dim = df_dim, name="conv1")
                h0 = batch_norm(h0, is_training=self.isTraining, scope="dNorm1")
                h0 = lrelu(h0)
                h0 = conv_cond_concat(h0, yb)

                # conv2
                h1 = conv2d(h0, output_dim = df_dim * 2, name="conv2")
                h1 = batch_norm(h1, is_training=self.isTraining, scope="dNorm2")
                h1 = lrelu(h1)
                h1 = conv_cond_concat(h1, yb)

                # conv3
                h2 = conv2d(h1, output_dim= df_dim * 4, name="conv3")
                h2 = batch_norm(h2, is_training=self.isTraining, scope="dNorm3")
                h2 = lrelu(h2)
                h2 = conv_cond_concat(h2, yb)

                # # conv4
                h3 = conv2d(h2, output_dim=df_dim * 8, name="conv4")
                h3 = batch_norm(h3, is_training=self.isTraining, scope="dNorm4")
                h3 = lrelu(h3)
                h3 = conv_cond_concat(h3, yb)

                # fc1
                n_b, n_h, n_w, n_f = [int(x) for x in h3.get_shape()]
                h4 = tf.reshape(h3, [n_b, n_h * n_w * n_f])
                h4 = fc(h4, n_h * n_w * n_f, scope='d_fc1')
            else:
                # This is based on the DCGAN tensorflow Architecture
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.labelSize])
                x = conv_cond_concat(z, yb)

                h0 = conv2d(x, self.cdim + self.labelSize, name='d_h0_conv', filt=[8, 8])
                h0 = lrelu(h0)
                h0 = conv_cond_concat(h0, yb)

                h1 = conv2d(h0, df_dim + self.labelSize, name='d_h1_conv')
                h1 = batch_norm(h1, is_training=self.isTraining, scope="dNorm1")
                h1 = lrelu(h1)
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat([h1, y], axis=1, name="concat_h1")

                h2 = fc(h1, 1024, 'd_h2_lin')
                h2 = lrelu(h2)
                h2 = batch_norm(h1, is_training=self.isTraining, scope="dNorm2")
                h2 = tf.concat([h2, y], axis=1, name="concat_h2")

                h3 = fc(h2, 1, 'd_h3_lin')

                h4 = tf.nn.sigmoid(h3)

        return h4

    def buildModel(self):
        # define variables
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.zdim], name="z")
        self.l = tf.placeholder(tf.float32, [self.batch_size, self.labelSize], name="label")

        img_dimensions = [self.imageSize[0], self.imageSize[1], self.cdim]
        self.g_real = tf.placeholder(tf.float32, [self.batch_size] + img_dimensions, name="images")


        ### GENERATORS ###
        self.g_fake = self.buildGenerator(self.z, self.l)
        self.g_sample = self.buildGenerator(self.z, self.l, reuse=True, isTraining=False)

        ### DISCRIMINATORS ###
        self.d_real = self.buildDiscriminator(self.g_real, self.l)
        self.d_fake = self.buildDiscriminator(self.g_fake, self.l, reuse=True)

        print("BUILT MODELS")

        # define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like (self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
        self.g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like (self.d_fake)))
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        print("DEFINED LOSS FUNCTIONS")

        # define optimizer
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        print("DEFINED OPTIMIZERS")

        tf.summary.scalar("d_loss_real"   ,self.d_loss_real)
        tf.summary.scalar("d_loss_fake"   ,self.d_loss_fake)
        tf.summary.scalar("d_loss"      ,self.d_loss)
        tf.summary.scalar("g_loss"      ,self.g_loss)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.save_folder: self.writer = tf.summary.FileWriter(self.save_folder, self.sess.graph)

        return

    def train(self, f_batch):
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
                return ((img * 255.) + 1) * 2
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

        # Pretrains discriminator net against some real examples.
        # pre_train = 0
        # while pre_train <= 2:
        #     batch_z = np.random.uniform(-1., +1., [self.batch_size, self.zdim])
        #     batch_images, batch_labels = f_batch(self.batch_size)
        #     feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images}
        #     _, d_loss, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_real, self.summary], feed_dict = feed_dict)
        #     pre_train += 1

        while True:
            step += 1

            batch_z = np.random.uniform(-1., +1., [self.batch_size, self.zdim])
            batch_images, batch_labels = f_batch(self.batch_size)

            if step % 5 == 1:
                feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images}
                _, d_loss, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_real, self.summary], feed_dict = feed_dict)
            else:
                # Update generator
                #_, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
                _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
                _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
                feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images}
                _, d_loss, g_fake, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_fake, self.g_real, self.summary], feed_dict = feed_dict)

            if step % 10 == 0:
                print ("{}: loss(D)={:.4f}, loss(G)={:.4f}; time/step = {:.2f} sec".format(step, d_loss, g_loss, time.time() - start))
                start = time.time()

            if step % 100 == 0:
                # Run models outputting images as training is run.
                self.writer.add_summary(summary, step)

                l0 = np.random.uniform(-1, +1, [self.batch_size, self.labelSize])
                #l1 = np.random.uniform(-1, +1, [self.batch_size, self.labelSize])
                l1 = np.array([np.random.binomial(1, 0.5, self.labelSize) for x in range(self.batch_size)])
                #l1 = np.array([[x % 2, x % 7, x % 12, x % 20, x % 20, x % 20, x % 30] for x in range(self.batch_size)])
                # l1 = np.array([[x % 2, x % 8] for x in range(self.batch_size)])
                #l1 = np.array([[0%2, 0%7] for x in range(self.batch_size)])
                z1 = np.random.uniform(-1, +1, [self.batch_size, self.zdim])
                z2 = np.random.uniform(-1, +1, [self.zdim])
                z2 = np.expand_dims(z2, axis=0)
                z2 = np.repeat(z2, repeats=self.batch_size, axis=0)

                g_image1 = self.sess.run(self.g_sample, feed_dict={self.z:z1, self.l:l0})
                g_image2 = self.sess.run(self.g_sample, feed_dict={self.z:batch_z, self.l:batch_labels})
                scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}d_real.png".format(step)), tileImage(g_real, [64, 64]))
                scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}d_fake1.png".format(step)), tileImage(g_image1, [64, 64]))
                scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}d_fake2.png".format(step)), tileImage(g_image2, [64, 64]))
                self.saver.save(self.sess,os.path.join(self.save_folder, "model.ckpt"), step)

            # if step % 1000 == 0:
            #     freeze_graph('Generator/sprite')