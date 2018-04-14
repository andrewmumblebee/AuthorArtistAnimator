from flask import Flask, jsonify, render_template, request, g
import tensorflow as tf
import scipy
import os
import time
import numpy as np
app = Flask(__name__)

def load_graph():
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(r'C:\Users\andrew\Documents\Root\Repos\CC\AAA\src\models\model.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def tileImage(imgs, size):
    d = int(np.sqrt(imgs.shape[0]-1))+1
    h, w = imgs.shape[1], imgs.shape[2]
    if (imgs.shape[3] in (3,4,1)):
        colour = imgs.shape[3]
        img = np.zeros((h * d, w * d, colour))
        for idx, image in enumerate(imgs):
            i = idx // d
            j = idx-i*d
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                        'must have dimensions: HxW or HxWx3 or HxWx4')

graph = load_graph()
z_ip = graph.get_tensor_by_name("prefix/z:0")
l_ip = graph.get_tensor_by_name("prefix/label:0")
y_op = graph.get_tensor_by_name('prefix/Generator/sprite:0')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(graph=graph,config=config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generator')
def generator():
    return render_template('generator.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/_generate_sprite')
def generate_sprite():
    #text = request.args.get('text', 'naked', type=str)
    z1 = np.random.uniform(-1, +1, [64, 100]) # Each time we could feed in a random noise vector to give some more randomness.
    l0 = np.array([[0%2, 0%8, 0%8, 0%8, 0%8, 0%8, 0%8] for x in range(64)])
    sprite = session.run(y_op, feed_dict={
        z_ip: z1,
        l_ip: l0
    })

    scipy.misc.imsave(r"C:\Users\andrew\Documents\Root\Repos\CC\AAA\web\static\images\sprite.png", tileImage(sprite, [64, 64]) * 255.)

    return jsonify(result=time.time())

    # with tf.Session() as sess:
    #     l0 = np.array([[0, 0, 0, 0, 0, 0, 0] for x in range(64)])
    #     z1 = np.random.uniform(-1, +1, [64, 100])
    #     saver = tf.train.import_meta_graph(r'C:\Users\andrew\Documents\Root\Repos\CC\AAA\src\models\model.ckpt-1300.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint(r'C:\Users\andrew\Documents\Root\Repos\CC\AAA\src\models'))
    #     z_op = sess.graph.get_operation_by_name("z")
    #     l_op = sess.graph.get_operation_by_name("label")
    #     g_image = sess.run('Generator', feed_dict={z_op: z1, l_op: l0})
    #     scipy.misc.imsave(r"C:\Users\andrew\Documents\Root\Repos\CC\AAA\web\static\imagessprite.png", tileImage(g_image, [64, 64]) * 255.)
    # # Do a prediction and output model#

if __name__ == '__main__':
    app.run()
