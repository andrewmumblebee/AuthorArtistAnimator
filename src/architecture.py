import tensorflow as tf
from operations import *

def discriminator(z, y, df_dim, c_dim, batch_size, labelSize, reuse=False, isTraining=True):
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

def animation_generator(z, y, i_dim, gf_dim, c_dim, batch_size, labelSize, reuse=False, isTraining=True):
    with tf.variable_scope("Generator") as scope:
        if reuse: scope.reuse_variables()

        s = i_dim[0]
        s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

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
        #h1 = tf.nn.dropout(h1, 0.5)
        h1 = tf.concat([h1, e5], axis=3, name="concat_h1")
        h1 = conv_cond_concat(h1, yb)

        h2 = deconv2d(h1, [batch_size, s32, s32, gf_dim * 8], name='g_h2')
        h2 = batch_norm(h2, is_training=isTraining, scope="gNorm8")
        h2 = tf.nn.relu(h2)
        #h2 = tf.nn.dropout(h2, 0.5)
        h2 = tf.concat([h2, e4], axis=3, name="concat_h2")
        h2 = conv_cond_concat(h2, yb)

        h3 = deconv2d(h2, [batch_size, s16, s16, gf_dim * 8], name='g_h3')
        h3 = batch_norm(h3, is_training=isTraining, scope="gNorm9")
        h3 = tf.nn.relu(h3)
        #h3 = tf.nn.dropout(h3, 0.5)
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