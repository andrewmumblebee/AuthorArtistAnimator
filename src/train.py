import argparse
from utility import BatchGenerator, ABatchGenerator
from model import Artist, Animator
import tensorflow as tf

if __name__=="__main__":
    flags = tf.app.flags
    parser = argparse.ArgumentParser()
    flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
    flags.DEFINE_integer("zdim", 10, "The noise vector size")
    flags.DEFINE_integer("cdim", 4, "Colour dimensions of image, 4 for alpha layer [4]")
    flags.DEFINE_integer("gf_dim", 64, "Colour dimensions of image, 4 for alpha layer [64]")
    flags.DEFINE_integer("df_dim", 64, "Colour dimensions of image, 4 for alpha layer [64]")
    flags.DEFINE_integer("image_height", 64, "The size of batch images [64]")
    flags.DEFINE_integer("image_width", 64, "The size of batch images [64]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_string("reload", None, "reload")
    flags.DEFINE_string("save_folder", "models", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("dataset_folder", r"C:\Users\andrew\Documents\Root\Repos\CC\AAA\src\data", "folder with images")
    flags.DEFINE_boolean("train", True, "True for training, False for testing [True]")
    flags.DEFINE_string("model", 'Artist', "Model to train Artist, Author [Artist]")
    flags.DEFINE_integer("epoch", 200, "Number of epochs to train for [25]")
    FLAGS = flags.FLAGS

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    with tf.Session(config=config) as sess:
        if FLAGS.model == "animator":
            print("\nTraining Animator Network")
            batch = ABatchGenerator(FLAGS.dataset_folder + r'\animations')
            label_size = batch.get_label_size()
            animator = Animator(sess=sess, isTraining=True, imageSize=[FLAGS.image_height, FLAGS.image_width], labelSize=label_size, args=FLAGS)
            animator.train(batch_generator=batch)
        else:
            print("\nTraining Artist Network")
            batch = BatchGenerator(FLAGS.dataset_folder + r'\sprites')
            label_size = batch.get_label_size()
            artist = Artist(sess=sess, isTraining=True, imageSize=[FLAGS.image_height, FLAGS.image_width], labelSize=label_size, args=FLAGS)
            artist.train(batch_generator=batch)
