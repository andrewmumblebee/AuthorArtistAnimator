# OPERATIONS MODULE - Containing useful operations that the models depend on.
import tensorflow as tf

def batch_norm(x, is_training=True, scope="batch_norm", epsilon=1e-5, decay=0.9):
    """ Pefroms batch normalization on the input shape.

        Args:
            x: Shape to have batch normalization applied to.
            is_training: Whether the graph is training, and statistics should be udpated.
            scope: Name of the scope in which the batch_normalization lies.
            epsilon: Small float added to variance to avoid diving by zero.
            decay: Decay for moving average.
    """
    return tf.contrib.layers.batch_norm(x,
                      decay=decay,
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=is_training,
                      scope=scope)

def deconv2d(input_, output_shape,
       filt = [5, 5], strides=[1, 2, 2, 1], stddev=0.02,
       name="deconv2d"):
    """ Performs a deconvolution on the input shape, based on supplied parameters.

        Just a transpose of a convolution operation.

        Args:
            - input_: Input shape to have a deconvolution operation applied to.
            - output_dim: Dimensions of the output shape created by the deconvolution.
            - filt: Dimensions of feature maps to be created. [5,5]
            - strides: Size of stride to take by (x,y) during deconvolution. [2, 2]
            - stddev: stddev of random normal initializer.
            - name: unique name to give node on graph.
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [filt[0], filt[1], output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.random_normal_initializer(stddev=stddev))

        # Perform a deconvolution on the input shape.
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=strides, padding = "SAME", data_format="NHWC")

        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), tf.shape(deconv))

        tf.summary.histogram("{}_w".format(name), w)
        tf.summary.histogram("{}_b".format(name), b)

        return deconv

def conv_cond_concat(x, y):
  """ Concatenate conditioning vector on feature map axis.

      Args:
        - x: feature maps.
        - y: conditioning vector.
  """
  x_shapes = tf.shape(x)
  y_shapes = tf.shape(y)
  y_ = y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])
  return tf.concat([x, y_], 3)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def conv2d(input_, output_dim, filt=[5, 5], strides=[2, 2],
           stddev=0.02, name="conv2d"):
  """ Performs a convolution on the input shape, based on supplied parameters.

      Args:
        - input_: Input shape to have a convolution operation applied to.
        - output_dim: Dimensions of the output shape created by the convolution.
        - filt: Dimensions of feature maps to be created. [5,5]
        - strides: Size of stride to take by (x,y) during convolution. [2, 2]
        - stddev: might be removed.
        - name: unique name to give node on graph.
  """
  with tf.variable_scope(name):
    w = tf.get_variable('w', [filt[0], filt[1], input_.get_shape()[-1], output_dim], #tf.truncated_normal_initializer(stddev=stddev)
              initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.nn.conv2d(input_, w, strides=[1, strides[0], strides[1], 1], padding='SAME')

    b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))

    tf.summary.histogram("{}_w".format(name), w)
    tf.summary.histogram("{}_b".format(name), b)

    return conv

def fc(input_, output_size, scope=None, stddev=0.02,
       bias_start=0.0, name='fc'):
    """ Creates a fully connected layer from the input shape to the given output size.

        Args:
            - input_: Input shape of the fully connected layer.
            - output_size: Dimensions of the created fully connected layer.
            - scope: Name of the scope in which the fully connected layer lies.
            - stddev: stddev of random normal initializer.
            - bias_start: initial value for biases.
            - name: unique name to give node on graph.
    """
    # Linear Fully connected layer
    with tf.variable_scope(scope or "FullyConnected"):
        w = tf.get_variable('w', [input_.get_shape()[1], output_size], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size],
        initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, w) + b

def freeze_graph(output_node_names, model_name, model_dir):
    """ Extract the sub graph defined by the output nodes and convert
        all its variables into constant, so the graph can be reused.

        Args:
            output_node_names: a string, containing all the output node's names,
                                comma separated
            model_name: unique name to give the model file.
            model_dir: the root folder containing the checkpoint state file
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # Retrieves latest checkpoint of the model.
    checkpoint = tf.train.latest_checkpoint(model_dir)
    absolute_model_dir = "\\".join(checkpoint.split('\\')[:-1])
    output_graph = absolute_model_dir + "\\" + model_name + "-model.pb"

    # Clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=clear_devices)
        saver.restore(sess, checkpoint)

        gd = sess.graph.as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # Retrieves the weights
            gd, # Retrieves the nodes
            output_node_names.split(",") # Output node names selects useful nodes
        )

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("Graph saved to path {}".format(output_graph))

    return output_graph_def
