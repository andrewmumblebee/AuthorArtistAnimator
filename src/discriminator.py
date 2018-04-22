import tensorflow as tf
from operations import *

def artist_discriminator(z, y, df_dim, c_dim, batch_size, labelSize, reuse=False, isTraining=True):
    with tf.variable_scope("Discriminator") as scope:
        if reuse: scope.reuse_variables()

        cgan_layer = True
        if cgan_layer:
            # This is based on the CGAN architecture

            # conditional layer
            yb = tf.reshape(y, [batch_size, 1, 1, labelSize])
            z = conv_cond_concat(z, yb)

            # conv1 CHANNELS + labelSize
            h0 = conv2d(z, output_dim = df_dim, name="conv1")
            h0 = batch_norm(h0, is_training=isTraining, scope="dNorm1")
            h0 = lrelu(h0)
            h0 = conv_cond_concat(h0, yb)

            # conv2
            h1 = conv2d(h0, output_dim = df_dim * 2, name="conv2")
            h1 = batch_norm(h1, is_training=isTraining, scope="dNorm2")
            h1 = lrelu(h1)
            h1 = conv_cond_concat(h1, yb)

            # conv3
            h2 = conv2d(h1, output_dim= df_dim * 4, name="conv3")
            h2 = batch_norm(h2, is_training=isTraining, scope="dNorm3")
            h2 = lrelu(h2)
            h2 = conv_cond_concat(h2, yb)

            # # conv4
            h3 = conv2d(h2, output_dim=df_dim * 8, name="conv4")
            h3 = batch_norm(h3, is_training=isTraining, scope="dNorm4")
            h3 = lrelu(h3)
            h3 = conv_cond_concat(h3, yb)

            # fc1
            n_b, n_h, n_w, n_f = [int(x) for x in h3.get_shape()]
            h4 = tf.reshape(h3, [n_b, n_h * n_w * n_f])
            h4 = fc(h4, n_h * n_w * n_f, scope='d_fc1')
        else:
            # This is based on the DCGAN tensorflow Architecture
            yb = tf.reshape(y, [batch_size, 1, 1, labelSize])
            x = conv_cond_concat(z, yb)

            h0 = conv2d(x, c_dim + labelSize, name='d_h0_conv', filt=[8, 8])
            h0 = lrelu(h0)
            h0 = conv_cond_concat(h0, yb)

            h1 = conv2d(h0, df_dim + labelSize, name='d_h1_conv')
            h1 = batch_norm(h1, is_training=isTraining, scope="dNorm1")
            h1 = lrelu(h1)
            h1 = tf.reshape(h1, [batch_size, -1])
            h1 = tf.concat([h1, y], axis=1, name="concat_h1")

            h2 = fc(h1, 1024, 'd_h2_lin')
            h2 = lrelu(h2)
            h2 = batch_norm(h1, is_training=isTraining, scope="dNorm2")
            h2 = tf.concat([h2, y], axis=1, name="concat_h2")

            h3 = fc(h2, 1, 'd_h3_lin')

            h4 = tf.nn.sigmoid(h3)

    return h4