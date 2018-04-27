from flask import Flask, jsonify, render_template, request, g, url_for, json
import tensorflow as tf
import scipy
import os
import time
import numpy as np
app = Flask(__name__)

def load_graph(path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def tileImage(imgs):
    d = imgs.shape[0]
    h, w = imgs.shape[1], imgs.shape[2]
    if (imgs.shape[3] in (3,4,1)):
        colour = imgs.shape[3]
        img = np.zeros((h, w * d, colour))
        for idx, image in enumerate(imgs):
            i = idx
            img[0:h, i * w:i * w + w, :] = image
        return ((img * 255.) + 1) * 2
    else:
        raise ValueError('in merge(images,size) images parameter '
                        'must have dimensions: HxW or HxWx3 or HxWx4')

def createAnimation(start_id, anim_count, frame_count, base_sprites):
    for a in range(anim_count):
        img_batch = []
        cnd_batch = []
        for f in range(frame_count):
            cnd_vector = np.zeros(16)
            cnd_vector[start_id + a] = 1
            img_batch.append(base_sprites[a])
            cnd_batch.append(np.append(cnd_vector, [f]))
        f_count = np.zeros((len(cnd_batch), 1))
        anim = animator.run(y_ap, feed_dict= {
            b_ap: img_batch,
            l_ap: cnd_batch,
            b_asize: f_count
        })
        output_anim = np.concatenate(([base_sprites[a]], anim))
        scipy.misc.imsave(r"C:\Users\andrew\Documents\Root\Repos\CC\AAA\web\static\images\animations\a" + str(a + start_id) + ".png", tileImage(output_anim))
    return output_anim

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

artist_graph = load_graph(r'C:\Users\andrew\Documents\Root\Repos\CC\AAA\src\models\Artist-model.pb')
z_ip = artist_graph.get_tensor_by_name("prefix/z:0")
l_ip = artist_graph.get_tensor_by_name("prefix/label:0")
y_op = artist_graph.get_tensor_by_name('prefix/Generator_1/sprite:0')
b_size = artist_graph.get_tensor_by_name("prefix/batch_size:0")
artist = tf.Session(graph=artist_graph,config=config)


animator_graph = load_graph(r'C:\Users\andrew\Documents\Root\Repos\CC\AAA\src\models\Animator-model.pb')
b_ap = animator_graph.get_tensor_by_name("prefix/base:0")
l_ap = animator_graph.get_tensor_by_name("prefix/label:0")
y_ap = animator_graph.get_tensor_by_name('prefix/Generator_1/sprite:0')
b_asize = animator_graph.get_tensor_by_name("prefix/batch_size:0")
animator = tf.Session(graph=animator_graph,config=config)


@app.route('/')
def index():
    json_path = os.path.join(app.root_path, "static\data", "encoding.json")
    data = json.load(open(json_path))

    animations = [[3, 7], [3, 8], [3, 9], [3, 6], [3, 13], [1, 6]] # Array storing the animations and their frame counts.
    return render_template('index.html', encoding = data, animations = animations)

@app.route('/_generate_sprite', methods=['POST', 'GET'])
def generate_sprite():
    combinations = request.get_json()
    input_ = [[0, 0, 1] + combinations, [1, 0, 0] + combinations, [0, 1, 0] + combinations]
    batch_size = np.zeros((len(input_), 1))
    #filler = np.repeat(np.array([combos[0]]), 60, axis = 0)
    #input_ = np.concatenate((combos, filler), axis = 0)

    #text = request.args.get('text', 'naked', type=str)
    #z1 = np.random.uniform(-1, +1, [64, 10]) # Each time we could feed in a random noise vector to give some more randomness.
    z1 = np.zeros([len(input_), 10])
    #l0 = np.random.uniform(-1, +1, [64, 72])
    #l0 = np.zeros([64, 72])
    #l0[0][0] = np.random.random_sample() // 0.5
    #l0[0][1] = 1
    #l0 = np.array([np.random.binomial(1, 0.1, 72) for x in range(64)])
    #sprites = np.concatenate((img1, img2), axis =1)

    sprites = artist.run(y_op, feed_dict={
        z_ip: z1,
        l_ip: input_,
        b_size: batch_size
    })

    anim_one = createAnimation(0, 3, 6, sprites)
    anim_two = createAnimation(3, 3, 7, sprites)
    anim_three = createAnimation(6, 3, 8, sprites)
    anim_four = createAnimation(9, 3, 5, sprites)
    anim_five = createAnimation(12, 3, 12, sprites)
    anim_six = createAnimation(15, 1, 5, sprites)

    print("Done with sprite")

    scipy.misc.imsave(r"C:\Users\andrew\Documents\Root\Repos\CC\AAA\web\static\images\sprite.png", tileImage(sprites))



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
