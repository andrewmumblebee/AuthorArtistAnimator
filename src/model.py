import os
import tensorflow as tf
import numpy as np
import argparse
import math, time
import scipy
from utility import BatchGenerator, tileImage
from operations import *
from generator import artist_generator, animation_generator
from discriminator import artist_discriminator, animator_discriminator

class GAN(object):
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        self.batch_size = args.batch_size
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

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

class Animator(GAN):
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        GAN.__init__(self, sess, isTraining, imageSize, labelSize, args)
        self.buildModel()
        return

    def buildModel(self):
        self.l = tf.placeholder(tf.float32, [self.batch_size, self.labelSize], name="label")

        img_dimensions = [self.imageSize[0], self.imageSize[1], self.cdim]
        self.z = tf.placeholder(tf.float32, [self.batch_size] + img_dimensions, name="base")
        self.g_real = tf.placeholder(tf.float32, [self.batch_size] + img_dimensions, name="images")

        ### GENERATORS ###
        self.g_fake = animation_generator(self.z, self.l, img_dimensions, 64, self.cdim, self.batch_size, self.labelSize)
        self.g_sample = animation_generator(self.z, self.l, img_dimensions, 64, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=False)

        ### DISCRIMINATORS ###
        self.d_real = animator_discriminator(self.z, self.l, self.g_real, 64, self.cdim, self.batch_size, self.labelSize, isTraining=self.isTraining)
        self.d_fake = animator_discriminator(self.z, self.l, self.g_fake, 64, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=self.isTraining)

        # define loss
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

        for epoch in range(self.epoch):

            batch_steps = batch_generator.get_file_count() // self.batch_size

            for step in range(batch_steps):

                batch_z = np.random.uniform(-1., +1., [self.batch_size, self.zdim])
                batch_images, batch_labels, batch_bases = batch_generator.get_batch(self.batch_size)
                # if step % 5 == 0:
                #     batch_labels = batch_labels * np.random.normal(0 , 1, [self.batch_size, self.labelSize])

                if step % 5 == 1:
                    feed_dict = {self.z : batch_bases, self.l : batch_labels, self.g_real : batch_images}
                    _, d_loss, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_real, self.summary], feed_dict = feed_dict)
                else:
                    # Update generator
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss], feed_dict={self.z: batch_bases, self.l: batch_labels, self.g_real: batch_images})
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss], feed_dict={self.z: batch_bases, self.l: batch_labels, self.g_real: batch_images})
                    feed_dict = {self.z : batch_bases, self.l : batch_labels, self.g_real : batch_images}
                    _, d_loss, g_fake, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_fake, self.g_real, self.summary], feed_dict = feed_dict)

                if step % 10 == 0:
                    print ("Epoch {}: [{}/{}] loss(D)={:.4f}, loss(G)={:.4f}; time/step = {:.2f} sec".format(epoch, step, batch_steps, d_loss, g_loss, time.time() - start))
                    start = time.time()

                if step % 100 == 0:
                    # Run models outputting images as training is run.
                    self.writer.add_summary(summary, step)
                    scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_bases.png".format(epoch, step)), tileImage(batch_bases, [64, 64]))
                    self.generate_sample(g_real, batch_z, batch_labels, epoch, step, batch_bases)

            batch_generator.reset_buffer()

            #freeze_graph('Generator/sprite')

    def generate_sample(self, real_image, batch_z, batch_labels, epoch, step, bases):
        # Run models outputting images as training is run.

        l0 = np.random.uniform(-1, +1, [self.batch_size, self.labelSize])
        l1 = np.array([np.random.binomial(1, 0.5, self.labelSize) for x in range(self.batch_size)])

        binomial_image = self.sess.run(self.g_sample, feed_dict={self.z:bases, self.l:l1})
        noise_image = self.sess.run(self.g_sample, feed_dict={self.z:bases, self.l:l0})
        matched_image = self.sess.run(self.g_sample, feed_dict={self.z:bases, self.l:batch_labels})

        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_real.png".format(epoch, step)), tileImage(real_image, [64, 64]))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_matched.png".format(epoch, step)), tileImage(matched_image, [64, 64]))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_noise.png".format(epoch, step)), tileImage(noise_image, [64, 64]))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_binomial.png".format(epoch, step)), tileImage(binomial_image, [64, 64]))

        self.saver.save(self.sess, os.path.join(self.save_folder, "model.ckpt"), step)


class Artist(GAN):
    """ Model for artist network, which learns how to draw sprites. """
    def __init__(self, sess, isTraining, imageSize, labelSize, args):
        GAN.__init__(self, sess, isTraining, imageSize, labelSize, args)
        self.buildModel()
        return

    def buildModel(self):
        # define variables
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.zdim], name="z")
        self.l = tf.placeholder(tf.float32, [self.batch_size, self.labelSize], name="label")

        img_dimensions = [self.imageSize[0], self.imageSize[1], self.cdim]
        self.g_real = tf.placeholder(tf.float32, [self.batch_size] + img_dimensions, name="images")


        ### GENERATORS ###
        self.g_fake = artist_generator(self.z, self.l, img_dimensions, 64, self.cdim, self.batch_size, self.labelSize)
        self.g_sample = artist_generator(self.z, self.l, img_dimensions, 64, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=False)

        ### DISCRIMINATORS ###
        self.d_real = artist_discriminator(self.g_real, self.l, 64, self.cdim, self.batch_size, self.labelSize, isTraining=self.isTraining)
        self.d_fake = artist_discriminator(self.g_fake, self.l, 64, self.cdim, self.batch_size, self.labelSize, reuse=True, isTraining=self.isTraining)

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

        for epoch in range(self.epoch):

            batch_steps = batch_generator.get_file_count() // self.batch_size

            for step in range(batch_steps):

                batch_z = np.random.uniform(-1., +1., [self.batch_size, self.zdim])
                batch_images, batch_labels = batch_generator.get_batch(self.batch_size)
                if step % 5 == 0:
                    batch_labels = batch_labels * np.random.normal(0 , 1, [self.batch_size, self.labelSize])

                if step % 5 == 1:
                    feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images}
                    _, d_loss, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_real, self.summary], feed_dict = feed_dict)
                else:
                    # Update generator
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
                    _, g_loss                = self.sess.run([self.g_optimizer, self.g_loss],feed_dict={self.z: batch_z, self.l: batch_labels})
                    feed_dict = {self.z : batch_z, self.l : batch_labels, self.g_real : batch_images}
                    _, d_loss, g_fake, g_real, summary = self.sess.run([self.d_optimizer, self.d_loss, self.g_fake, self.g_real, self.summary], feed_dict = feed_dict)

                if step % 10 == 0:
                    print ("Epoch {}: [{}/{}] loss(D)={:.4f}, loss(G)={:.4f}; time/step = {:.2f} sec".format(epoch, step, batch_steps, d_loss, g_loss, time.time() - start))
                    start = time.time()

                if step % 100 == 0:
                    # Run models outputting images as training is run.
                    self.writer.add_summary(summary, step)
                    self.generate_sample(g_real, batch_z, batch_labels, epoch, step)

            batch_generator.reset_buffer()

            freeze_graph('Generator/sprite')

    def generate_sample(self, real_image, batch_z, batch_labels, epoch, step):
        # Run models outputting images as training is run.

        l0 = np.random.uniform(-1, +1, [self.batch_size, self.labelSize])
        #l1 = np.random.uniform(-1, +1, [self.batch_size, self.labelSize])
        l1 = np.array([np.random.binomial(1, 0.5, self.labelSize) for x in range(self.batch_size)])
        #l1 = np.array([[x % 2, x % 7, x % 12, x % 20, x % 20, x % 20, x % 30] for x in range(self.batch_size)])
        #l1 = np.array([[x % 2, x % 8] for x in range(self.batch_size)])
        #l1 = np.array([[0%2, 0%7] for x in range(self.batch_size)])
        z1 = np.random.uniform(-1, +1, [self.batch_size, self.zdim])
        z2 = np.random.uniform(-1, +1, [self.zdim])
        z2 = np.expand_dims(z2, axis=0)
        z2 = np.repeat(z2, repeats=self.batch_size, axis=0)

        binomial_image = self.sess.run(self.g_sample, feed_dict={self.z:z1, self.l:l1})
        noise_image = self.sess.run(self.g_sample, feed_dict={self.z:z2, self.l:l0})
        matched_image = self.sess.run(self.g_sample, feed_dict={self.z:batch_z, self.l:batch_labels})

        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_real.png".format(epoch, step)), tileImage(real_image, [64, 64]))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_matched.png".format(epoch, step)), tileImage(matched_image, [64, 64]))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_noise.png".format(epoch, step)), tileImage(noise_image, [64, 64]))
        scipy.misc.imsave(os.path.join(self.save_folder,"images","img_{}_{}_binomial.png".format(epoch, step)), tileImage(binomial_image, [64, 64]))

        self.saver.save(self.sess, os.path.join(self.save_folder, "model.ckpt"), step)