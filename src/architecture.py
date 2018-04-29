import tensorflow as tf
from operations import *

def discriminator(z, y, df_dim, c_dim, batch_size, labelSize, reuse=False, isTraining=True):
    """ Discriminator model, used by both the artist and animator systems.

        Args:
            - z: noise vector.
            - y: conditioning vector.
            - df_dim: feature map filter dimensions.
            - c_dim: colour dimensions of img.
            - batch_size: batch size of images being fed in.
            - labelSize: length of conditioning vector.
            - reuse: Toggles reusing variables in same scope.
            - isTraining: Whether to update batch_norm statistics as data is fed in.
    """
    with tf.variable_scope("Discriminator") as scope:
        if reuse: scope.reuse_variables()

        # This is based on the CGAN architecture
        batch_size = tf.shape(batch_size)[0]

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

        # conv4
        h3 = conv2d(h2, output_dim=df_dim * 8, name="conv4")
        h3 = batch_norm(h3, is_training=isTraining, scope="dNorm4")
        h3 = lrelu(h3)
        h3 = conv_cond_concat(h3, yb)

        # fc1
        r_shape = tf.shape(h3)
        f_shape = h3.get_shape()
        h4 = tf.reshape(h3, [r_shape[0], f_shape[1] * f_shape[2] * f_shape[3]])
        h4 = fc(h4, f_shape[1] * f_shape[2] * f_shape[3], scope='d_fc1')

    return h4

def artist_generator(z, y, i_dim, gf_dim, c_dim, b_size, labelSize, reuse=False, isTraining=True):
    """ Creates the artist generator model.

        Args:
            - z: noise vector.
            - y: conditioning vector.
            - i_dim: image dimensions.
            - gf_dim: feature map filter dimensions.
            - c_dim: colour dimensions of img.
            - b_size: batch size of images being fed in.
            - labelSize: length of conditioning vector.
            - reuse: Toggles reusing variables in same scope.
            - isTraining: Whether to update batch_norm statistics as data is fed in.
    """
    with tf.variable_scope("Generator") as scope:
        if reuse: scope.reuse_variables()

        s_h, s_w = i_dim[0], i_dim[1]
        s_h2, s_h4, s_h8 = int(s_h/2), int(s_h/4), int(s_h/8)
        s_w2, s_w4, s_w8 = int(s_w/2), int(s_w/4), int(s_w/8)
        batch_size = tf.shape(b_size)[0]

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

def animator_generator(z, y, i_dim, gf_dim, c_dim, b_size, labelSize, reuse=False, isTraining=True):
    """ Creates the animator generator model.

        Args:
            - z: noise vector.
            - y: conditioning vector.
            - i_dim: image dimensions.
            - gf_dim: feature map filter dimensions.
            - c_dim: colour dimensions of img.
            - b_size: batch size of images being fed in.
            - labelSize: length of conditioning vector.
            - reuse: Toggles reusing variables in same scope.
            - isTraining: Whether to update batch_norm statistics as data is fed in.
    """
    with tf.variable_scope("Generator") as scope:
        if reuse: scope.reuse_variables()

        s = i_dim[0]
        s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)
        batch_size = tf.shape(b_size)[0]

        yb = tf.reshape(y, [batch_size, 1, 1, labelSize])
        z = conv_cond_concat(z, yb)

        e0 = conv2d(z, output_dim = gf_dim, name="conv0")
        e0 = batch_norm(e0, is_training=isTraining, scope="dNorm0")
        e0 = lrelu(e0)
        e0 = conv_cond_concat(e0, yb)

        e1 = conv2d(e0, output_dim = gf_dim * 4, name="conv1")
        e1 = batch_norm(e1, is_training=isTraining, scope="dNorm1")
        e1 = lrelu(e1)
        e1 = conv_cond_concat(e1, yb)

        e2 = conv2d(e1, output_dim = gf_dim * 8, name="conv2")
        e2 = batch_norm(e2, is_training=isTraining, scope="dNorm2")
        e2 = lrelu(e2)
        e2 = conv_cond_concat(e2, yb)

        e3 = conv2d(e2, output_dim = gf_dim * 8, name="conv3")
        e3 = batch_norm(e3, is_training=isTraining, scope="dNorm3")
        e3 = lrelu(e3)
        e3 = conv_cond_concat(e3, yb)

        e4 = conv2d(e3, output_dim = gf_dim * 8, name="conv4")
        e4 = batch_norm(e4, is_training=isTraining, scope="dNorm4")
        e4 = lrelu(e4)
        e4 = conv_cond_concat(e4, yb)

        e5 = conv2d(e4, output_dim = gf_dim * 8, name="conv5")
        e5 = batch_norm(e5, is_training=isTraining, scope="dNorm5")
        e5 = lrelu(e5)
        e5 = conv_cond_concat(e5, yb)

        e6 = conv2d(e5, output_dim = gf_dim * 8, name="conv6")
        e6 = batch_norm(e6, is_training=isTraining, scope="dNorm6")
        e6 = lrelu(e6)
        e6 = conv_cond_concat(e6, yb)

        h1 = deconv2d(e6, [batch_size, s64, s64, gf_dim * 8], name='g_h1')
        h1 = batch_norm(h1, is_training=isTraining, scope="gNorm7")
        h1 = tf.nn.relu(h1)
        h1 = tf.concat([h1, e5], axis=3, name="concat_h1")
        h1 = conv_cond_concat(h1, yb)

        h2 = deconv2d(h1, [batch_size, s32, s32, gf_dim * 8], name='g_h2')
        h2 = batch_norm(h2, is_training=isTraining, scope="gNorm8")
        h2 = tf.nn.relu(h2)
        h2 = tf.concat([h2, e4], axis=3, name="concat_h2")
        h2 = conv_cond_concat(h2, yb)

        h3 = deconv2d(h2, [batch_size, s16, s16, gf_dim * 8], name='g_h3')
        h3 = batch_norm(h3, is_training=isTraining, scope="gNorm9")
        h3 = tf.nn.relu(h3)
        h3 = tf.concat([h3, e3], axis=3, name="concat_h3")
        h3 = conv_cond_concat(h3, yb)

        h4 = deconv2d(h3, [batch_size, s8, s8, gf_dim * 4], name='g_h4')
        h4 = batch_norm(h4, is_training=isTraining, scope="gNorm10")
        h4 = tf.nn.relu(h4)
        h4 = tf.concat([h4, e2], axis=3, name="concat_h4")
        h4 = conv_cond_concat(h4, yb)

        h5 = deconv2d(h4, [batch_size, s4, s4, gf_dim * 2], name='g_h5')
        h5 = batch_norm(h5, is_training=isTraining, scope="gNorm11")
        h5 = tf.nn.relu(h5)
        h5 = tf.concat([h5, e1], axis=3, name="concat_h5")
        h5 = conv_cond_concat(h5, yb)

        h6 = deconv2d(h5, [batch_size, s2, s2, gf_dim], name='g_h6')
        h6 = batch_norm(h6, is_training=isTraining, scope="gNorm4")
        h6 = tf.nn.relu(h6)
        h6 = tf.concat([h6, e0], axis=3, name="concat_h6")
        h6 = conv_cond_concat(h6, yb)

        return tf.nn.tanh(
            deconv2d(h6, [batch_size, s, s, c_dim], name='g_h7'), name='sprite')