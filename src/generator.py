from utility import calcImageSize
from operations import *
import tensorflow as tf

def artist_generator(z, y, i_dim, gf_dim, c_dim, batch_size, labelSize, reuse=False, isTraining=True):
    gf_dim = 64
    dim_0_h, dim_0_w = i_dim[0], i_dim[1] # Building backwards convolutional layers
    dim_1_h, dim_1_w = calcImageSize(dim_0_h, dim_0_w, stride=2)
    dim_2_h, dim_2_w = calcImageSize(dim_1_h, dim_1_w, stride=2)
    dim_3_h, dim_3_w = calcImageSize(dim_2_h, dim_2_w, stride=2)
    dim_4_h, dim_4_w = calcImageSize(dim_3_h, dim_3_w, stride=2)
    dim_5_h, dim_5_w = calcImageSize(dim_4_h, dim_4_w, stride=2)

    with tf.variable_scope("Generator") as scope:
        if reuse: scope.reuse_variables()

        cgan = False
        if cgan:
            yb = tf.reshape(y, [batch_size, 1, 1, labelSize]) # I.e. reshapes to [64, 1, 1, 74]
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

            h0 = deconv2d(h0, output_shape=[batch_size, dim_4_h, dim_4_w, gf_dim * 8], name="deconv5")
            h0 = batch_norm(h0, is_training=isTraining, scope="gNorm6")
            h0 = tf.nn.relu(h0)
            h0 = conv_cond_concat(h0, yb)

            # Deconv 1
            h0 = deconv2d(h0, output_shape=[batch_size, dim_3_h, dim_3_w, gf_dim * 4], name="deconv4")
            h0 = batch_norm(h0, is_training=isTraining, scope="gNorm4")
            h0 = tf.nn.relu(h0)
            h0 = conv_cond_concat(h0, yb)

            # Deconv 1
            h1 = deconv2d(h0, output_shape=[batch_size, dim_2_h, dim_2_w, gf_dim * 2], name="deconv3")
            h1 = batch_norm(h1, is_training=isTraining, scope="gNorm3")
            h1 = tf.nn.relu(h1)
            h1 = conv_cond_concat(h1, yb)

            # Deconv 2
            h2 = deconv2d(h1, output_shape = [batch_size, dim_1_h, dim_1_w, gf_dim * 1], name="deconv2", filt=[8, 8])
            h2 = batch_norm(h2, is_training=isTraining, scope="gNorm2")
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            # Deconv 3 -> 64x64 Image
            h3 = deconv2d(h2, output_shape=[batch_size, dim_0_h, dim_0_w, c_dim], name= "deconv1")

            # Using tanh as per improved DCGAN paper.
            y = tf.nn.tanh(h3, name="sprite")
        else:
            s_h, s_w = i_dim[0], i_dim[1]
            s_h2, s_h4, s_h8 = int(s_h/2), int(s_h/4), int(s_h/8)
            s_w2, s_w4, s_w8 = int(s_w/2), int(s_w/4), int(s_w/8)

            yb = tf.reshape(y, [batch_size, 1, 1, labelSize])
            z = tf.concat([z, y], axis=1, name="concat_z")

            h0 = fc(z, 1024, 'g_h0_lin')
            h0 = batch_norm(h0, is_training=isTraining, scope="gNorm1")
            h0 = tf.nn.relu(h0)
            h0 = tf.concat([h0, y], axis=1, name="concat_h1")

            h1 = fc(h0, 64 * 2 * s_h8 * s_w8, 'g_h1_lin')
            h1 = batch_norm(h1, is_training=isTraining, scope="gNorm2")
            h1 = tf.nn.relu(h1)
            h1 = tf.reshape(h1, [batch_size, s_h8, s_w8, 64 * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = deconv2d(h1, [batch_size, s_h4, s_w4, 64 * 2], name='g_h1')
            h2 = batch_norm(h2, is_training=isTraining, scope="gNorm3")
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            h2 = deconv2d(h2,[batch_size, s_h2, s_w2, 64 * 2], name='g_h2')
            h2 = batch_norm(h2, is_training=isTraining, scope="gNorm4")
            h2 = tf.nn.relu(h2)
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.tanh(
                deconv2d(h2, [batch_size, s_h, s_w, c_dim], name='g_h3'), name='sprite')

    return y