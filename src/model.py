import os
import tensorflow as tf
import numpy as np
import argparse
import math, time
import scipy
from utility import BatchGenerator, tileImage
from operations import *
from architecture import discriminator, artist_generator, animation_generator

class GAN(object):
    """ Base class of GAN.

        Sets attributes that are shared across both GAN models.
    """
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        self.bs = args.batch_size
        self.learning_rate = args.learning_rate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.save_folder = args.save_folder
        self.reload = args.reload
        self.epoch = args.epoch
        self.cdim = args.cdim
        self.labelSize = labelSize
        self.sess = sess
        self.gf_dim = args.gf_dim
        self.df_dim = args.df_dim

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

class Animator(GAN):
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        GAN.__init__(self, sess, isTraining, imageSize, labelSize, args)
        self.buildModel()
        return

    def buildModel(self):
        self.batch_size = tf.placeholder(tf.int32, [None, 1], name="batch_size") # Enable dynamic batch size.
        self.l = tf.placeholder(tf.float32, [self.batch_size.get_shape()[0], self.labelSize], name="label")

        img_dimensions = [self.imageSize[0], self.imageSize[1], self.cdim]
        self.z = tf.placeholder(tf.float32, [self.batch_size.get_shape()[0]] + img_dimensions, name="base")
        self.g_real = tf.placeholder(tf.float32, [self.batch_size.get_shape()[0]] + img_dimensions, name="images")

        ### GENERATORS ###
        self.g_fake = animation_generator(self.z, self.l, img_dimensions, self.gf_dim, self.cdim, self.batch_size, self.labelSize)
        self.g_sample = animation_generator(self.z, self.l, img_dimensions, self.gf_dim, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=False)

        ### DISCRIMINATORS ###
        self.d_real = discriminator(self.z, self.l, self.df_dim, self.cdim, self.batch_size, self.labelSize, isTraining=self.isTraining)
        self.d_fake = discriminator(self.z, self.l, self.df_dim, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=self.isTraining)

        # Define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like (self.d_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
        self.g_loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like (self.d_fake))) \
                            + 100 * tf.reduce_mean(tf.abs(self.g_real - self.g_fake))
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.g_loss, var_list=[x for x in tf.trainable_variables() if "Generator"     in x.name])
        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.d_loss, var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.save_folder: self.writer = tf.summary.FileWriter(self.save_folder, self.sess.graph)

    def train(self, batch_generator):
        if self.save_folder and not os.path.exists(os.path.join(self.save_folder,"images")):
            os.makedirs(os.path.join(self.save_folder,"images"))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.loadModel(self.reload)
        start = time.time()
        self.batch_s = np.zeros((self.bs, 1))

        for epoch in range(self.epoch):

            batch_steps = batch_generator.get_file_count() // self.bs

            for step in range(batch_steps):

                batch_z = np.random.uniform(-1., +1., [self.bs, self.zdim])
                batch_images, batch_labels, batch_bases = batch_generator.get_batch(self.bs)
                # if step % 5 == 0:
                #     batch_labels = batch_labels * np.random.normal(0 , 1, [self.batch_size, self.labelSize])

                if step % 5 == 1:
                    feed_dict = {self.z : batch_bases, self.l : batch_labels, self.g_real : batch_images, self.batch_size: self.batch_s}
                    _, d_loss, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_real, self.summary], feed_dict = feed_dict)
                else:
                    # Update generator
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss], feed_dict={self.z: batch_bases, self.l: batch_labels, self.g_real: batch_images, self.batch_size: self.batch_s})
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss], feed_dict={self.z: batch_bases, self.l: batch_labels, self.g_real: batch_images, self.batch_size: self.batch_s})
                    feed_dict = {self.z : batch_bases, self.l : batch_labels, self.g_real : batch_images, self.batch_size: self.batch_s}
                    _, d_loss, g_fake, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_fake, self.g_real, self.summary], feed_dict = feed_dict)

                if step % 10 == 0:
                    print ("Epoch {}: [{}/{}] loss(D)={:.4f}, loss(G)={:.4f}; time/step = {:.2f} sec".format(epoch, step, batch_steps, d_loss, g_loss, time.time() - start))
                    start = time.time()

                if step % 100 == 0:
                    # Run models outputting images as training is run.
                    self.writer.add_summary(summary, step)
                    scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_bases.png".format(epoch, step)), tileImage(batch_bases))
                    self.generate_sample(g_real, batch_z, batch_labels, epoch, step, batch_bases)

            batch_generator.reset_buffer()

            freeze_graph('Generator_1/sprite', 'Animator')

    def generate_sample(self, real_image, batch_z, batch_labels, epoch, step, bases):
        # Run models outputting images as training is run.

        l0 = np.random.uniform(-1, +1, [self.bs, self.labelSize])
        l1 = np.array([np.random.binomial(1, 0.5, self.labelSize) for x in range(self.bs)])

        binomial_image = self.sess.run(self.g_sample, feed_dict={self.z:bases, self.l:l1, self.batch_size: self.batch_s})
        noise_image = self.sess.run(self.g_sample, feed_dict={self.z:bases, self.l:l0, self.batch_size: self.batch_s})
        matched_image = self.sess.run(self.g_sample, feed_dict={self.z:bases, self.l:batch_labels, self.batch_size: self.batch_s})

        scipy.misc.imsave(os.path.join(self.save_folder,"images","anim_img_{}_{}_real.png".format(epoch, step)), tileImage(real_image))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","anim_img_{}_{}_matched.png".format(epoch, step)), tileImage(matched_image))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","anim_img_{}_{}_noise.png".format(epoch, step)), tileImage(noise_image))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","anim_img_{}_{}_binomial.png".format(epoch, step)), tileImage(binomial_image))

        self.saver.save(self.sess, os.path.join(self.save_folder, "model.ckpt"), step)


class Artist(GAN):
    """ Model for artist network, which learns how to draw sprites. """
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        GAN.__init__(self, sess, isTraining, imageSize, labelSize, args)
        self.buildModel()
        return

    def buildModel(self):
        # define variables
        self.batch_size = tf.placeholder(tf.int32, [None, 1], name="batch_size")
        self.z = tf.placeholder(tf.float32, [self.batch_size.get_shape()[0], self.zdim], name="z")
        self.l = tf.placeholder(tf.float32, [self.batch_size.get_shape()[0], self.labelSize], name="label")

        img_dimensions = [self.imageSize[0], self.imageSize[1], self.cdim]
        self.g_real = tf.placeholder(tf.float32, [self.batch_size.get_shape()[0]] + img_dimensions, name="images")


        ### GENERATORS ###
        self.g_fake = artist_generator(self.z, self.l, img_dimensions, self.gf_dim, self.cdim, self.batch_size, self.labelSize)
        self.g_sample = artist_generator(self.z, self.l, img_dimensions, self.gf_dim, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=False)

        ### DISCRIMINATORS ###
        self.d_real = discriminator(self.g_real, self.l, self.df_dim, self.cdim, self.batch_size, self.labelSize, isTraining=self.isTraining)
        self.d_fake = discriminator(self.g_fake, self.l, self.df_dim, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=self.isTraining)

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

    def train(self, batch_generator):
        if self.save_folder and not os.path.exists(os.path.join(self.save_folder,"images")):
            os.makedirs(os.path.join(self.save_folder,"images"))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.loadModel(self.reload)
        start = time.time()
        self.batch_s = np.zeros((self.bs, 1))

        for epoch in range(self.epoch):

            batch_steps = batch_generator.get_file_count() // self.bs

            for step in range(batch_steps):

                batch_z = np.random.uniform(-1., +1., [self.bs, self.zdim])
                batch_images, batch_labels = batch_generator.get_batch(self.bs)

                # Add some random noise to the labels every 5 steps, to train GAN to generalize.
                if step % 5 == 0:
                    batch_labels = batch_labels * np.random.uniform(0, 1, [self.bs, self.labelSize])

                feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images, self.batch_size : self.batch_s}

                # Every now and again train discriminator model more.
                if step % 5 == 1:
                    _, d_loss, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_real, self.summary], feed_dict = feed_dict)
                else:
                    # Update generator
                    _, g_loss              = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels, self.batch_size : self.batch_s})
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels, self.batch_size : self.batch_s})
                    _, d_loss, g_fake, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_fake, self.g_real, self.summary], feed_dict = feed_dict)

                if step % 10 == 0:
                    print ("Epoch {}: [{}/{}] loss(D)={:.4f}, loss(G)={:.4f}; time/step = {:.2f} sec".format(epoch, step, batch_steps, d_loss, g_loss, time.time() - start))
                    start = time.time()

                if step % 100 == 0:
                    # Run models outputting images as training is run.
                    self.writer.add_summary(summary, step)
                    self.generate_sample(g_real, batch_z, batch_labels, epoch, step)

            freeze_graph('Generator_1/sprite', 'Artist')

            batch_generator.reset_buffer()


    def generate_sample(self, real_image, batch_z, batch_labels, epoch, step):
        # Run models outputting images as training is run.

        l0 = np.random.uniform(-1, +1, [self.bs, self.labelSize])
        l1 = np.array([np.random.binomial(1, 0.5, self.labelSize) for x in range(self.bs)])
        z1 = np.random.uniform(-1, +1, [self.bs, self.zdim])

        binomial_image = self.sess.run(self.g_sample, feed_dict={self.z:z1, self.l:l1, self.batch_size : self.batch_s})
        noise_image = self.sess.run(self.g_sample, feed_dict={self.z:z1, self.l:l0, self.batch_size : self.batch_s})
        matched_image = self.sess.run(self.g_sample, feed_dict={self.z:batch_z, self.l:batch_labels, self.batch_size : self.batch_s})

        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_real.png".format(epoch, step)), tileImage(real_image))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_matched.png".format(epoch, step)), tileImage(matched_image))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_noise.png".format(epoch, step)), tileImage(noise_image))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_binomial.png".format(epoch, step)), tileImage(binomial_image))

        self.saver.save(self.sess, os.path.join(self.save_folder, "model.ckpt"), step)