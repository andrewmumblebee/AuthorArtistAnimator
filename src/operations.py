import tensorflow as tf

def batch_norm(x, is_training=True, scope="batch_norm", epsilon=1e-5, decay=0.9):
    return tf.contrib.layers.batch_norm(x,
                      decay=decay,
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=is_training,
                      scope=scope)

def deconv2d(input_, output_shape,
       filt = [5, 5], strides=[1, 2, 2, 1], stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [filt[0], filt[1], output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.random_normal_initializer(stddev=stddev))

        # Perform a deconvolution on the input shape.
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=strides, padding = "SAME", data_format="NHWC")

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape() # Conditioning vector with conditional properties
  return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv2d(input_, output_shape,
       filt=[5, 5], strides=[1, 2, 2, 1], stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', output_shape,
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME')

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias