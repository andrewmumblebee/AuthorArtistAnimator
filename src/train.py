""" Training module, use this to train the models on a given dataset. """
import argparse
from utility import BatchGenerator, AnimatorBatchGenerator
from model import Artist, Animator
import tensorflow as tf

if __name__=="__main__":
    flags = tf.app.flags
    parser = argparse.ArgumentParser()
    flags.DEFINE_integer("batch_size", 64, "Batch size to train with [64]")
    flags.DEFINE_integer("zdim", 10, "The noise vector size")
    flags.DEFINE_integer("cdim", 4, "Colour dimensions of image, 4 for alpha layer [4]")
    flags.DEFINE_integer("gf_dim", 64, "Dimensions of initial generator filter [64]")
    flags.DEFINE_integer("df_dim", 64, "Dimensions of initial discriminator filter [64]")
    flags.DEFINE_integer("image_height", 64, "Height of batch images [64]")
    flags.DEFINE_integer("image_width", 64, "Width of batch images [64]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_string("reload", None, "reload")
    flags.DEFINE_string("save_folder", "models", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("dataset_folder", "data", "Folder which contains data for training")
    flags.DEFINE_boolean("train", True, "True for training, False for testing [True]")
    flags.DEFINE_string("model", "Artist", "Model to train Artist, Author [Artist]")
    flags.DEFINE_integer("epoch", 25, "Number of epochs to train for [25]")
    FLAGS = flags.FLAGS

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    with tf.Session(config=config) as sess:
        """ Start training of a given network. """
        if FLAGS.model == "animator":
            print("\nTraining Animator Network")
            batch = AnimatorBatchGenerator(FLAGS.dataset_folder + r'\animations')
            label_size = batch.get_label_size()
            animator = Animator(sess=sess, isTraining=True, imageSize=[FLAGS.image_height, FLAGS.image_width], labelSize=label_size, args=FLAGS)
            animator.train(batch_generator=batch)
        else:
            print("\nTraining Artist Network")
            batch = BatchGenerator(FLAGS.dataset_folder + r'\sprites')
            label_size = batch.get_label_size()
            artist = Artist(sess=sess, isTraining=True, imageSize=[FLAGS.image_height, FLAGS.image_width], labelSize=label_size, args=FLAGS)
            artist.train(batch_generator=batch)
