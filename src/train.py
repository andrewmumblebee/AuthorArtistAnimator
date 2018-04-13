import argparse
from model import BatchGenerator, CGAN
import tensorflow as tf

if __name__=="__main__":
    flags = tf.app.flags
    parser = argparse.ArgumentParser()
    flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
    flags.DEFINE_integer("zdim", 100, "The size of batch images [64]")
    flags.DEFINE_integer("image_height", 64, "The size of batch images [64]")
    flags.DEFINE_integer("image_width", 64, "The size of batch images [64]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_string("reload", None, "reload")
    flags.DEFINE_string("save_folder", "models", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("dataset_folder", r'C:\Users\andrew\Documents\Root\Repos\CC\AAA\CharacterScraper\dump\sprites', "folder with images")
    flags.DEFINE_boolean("train", True, "True for training, False for testing [True]")
    FLAGS = flags.FLAGS

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        batch = BatchGenerator(FLAGS.dataset_folder)
        gan = CGAN(sess=sess, isTraining=True, imageSize=[FLAGS.image_height, FLAGS.image_width], labelSize=7, args=FLAGS)
        gan.train(f_batch=batch.getBatch)
