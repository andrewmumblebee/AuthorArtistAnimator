""" SERVER MODULE

    - Responsible for loading the models and running them.
    - When the model is run the output images are saved in the static folder.
    - Using cachebusting (Which is bad), the images are automatically updated on the webpage.
"""
from flask import Flask, jsonify, render_template, request, url_for, json
import tensorflow as tf
import scipy
import os
import time
import numpy as np
app = Flask(__name__)

def load_graph(path):
    """ Loads in protbuf file of graph and parse it to retrieve the graph_def.

        Args:
            - path[string]: path of the graph.pb file.
    """
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Imports the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="model")
    return graph

def joinImages(imgs):
    """ Joins a set of images into a single line of images.

        Args:
            - imgs[array]: numpy array, containing images in form [?, 64, 64, 4]
    """
    d = imgs.shape[0]
    h, w = imgs.shape[1], imgs.shape[2]
    colour = imgs.shape[3]
    img = np.zeros((h, w * d, colour))
    for idx, image in enumerate(imgs):
        i = idx
        img[0:h, i * w:i * w + w, :] = image
    return ((img * 255.) + 1) * 2

def createAnimation(start_id, anim_count, frame_count, base_sprites):
    """ Create a set of animations using the animator graph.

        This is used for batch generation of the animations, based on animations of the same type.
        i.e. shooting bow, facing forwards, backwards and to the side.

        Args:
            - start_id[int]: id to assign the output file, increases for each animation.
            - anim_count[int]: amount of animations to generate.
            - frame_count[int]: frame count of the animations, should be same for each in this batch.
            - base_sprites[list]: Images containing the base frames of the animation. (Created by Artist)
    """
    for a in range(anim_count):
        img_batch = []
        cnd_batch = []

        for f in range(frame_count):
            # Attaches encodings for each frame of the animation.
            cnd_vector = np.zeros(16)
            cnd_vector[start_id + a] = 1
            img_batch.append(base_sprites[a])
            cnd_batch.append(np.append(cnd_vector, [f]))

        f_count = np.zeros((len(cnd_batch), 1)) # Animation's frame count.

        # Creates a batch of images for one animation.
        anim = animator.run(y_ap, feed_dict= {
            b_ap: img_batch,
            l_ap: cnd_batch,
            b_asize: f_count
        })
        output_anim = np.concatenate(([base_sprites[a]], anim)) # Add base image to the output animation file.
        scipy.misc.imsave(app.root_path + "/static/images/animations/a" + str(a + start_id) + ".png", joinImages(output_anim))

    return output_anim

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

artist_graph = load_graph(app.root_path + "/../src/models/Artist-model.pb")
z_ip = artist_graph.get_tensor_by_name("model/z:0")
l_ip = artist_graph.get_tensor_by_name("model/label:0")
y_op = artist_graph.get_tensor_by_name("model/Generator_1/sprite:0")
b_size = artist_graph.get_tensor_by_name("model/batch_size:0")
artist = tf.Session(graph=artist_graph,config=config)

animator_graph = load_graph(app.root_path + "/../src/models/Animator-model.pb")
b_ap = animator_graph.get_tensor_by_name("model/base:0")
l_ap = animator_graph.get_tensor_by_name("model/label:0")
y_ap = animator_graph.get_tensor_by_name("model/Generator_1/sprite:0")
b_asize = animator_graph.get_tensor_by_name("model/batch_size:0")
animator = tf.Session(graph=animator_graph,config=config)


@app.route('/')
def index():
    """ Loads encodings and passes to index to fill out sliders and animation img placeholders. """
    json_path = os.path.join(app.root_path, "static/data", "encoding.json")
    data = json.load(open(json_path))

    # This is fairly odd way to do it, but it's in the form of [animCount, frameCount]
    # This allows the page to be filled automatically when loaded, to account for each type of animation.
    animations = [[3, 7], [3, 8], [3, 9], [3, 6], [3, 13], [1, 6]] # Array storing the animations and their frame counts.
    return render_template('index.html', encoding = data, animations = animations)

@app.route('/_generate_sprite', methods=['POST', 'GET'])
def generate_sprite():
    """ Callback from JSON request on index, creates sprites and updates html with them. """
    combinations = request.get_json() # Retrieves combinations from sliders.

    # Attach pose encodings to the combinations that were selected.
    input_ = [[0, 0, 1] + combinations, [1, 0, 0] + combinations, [0, 1, 0] + combinations]
    batch_size = np.zeros((len(input_), 1)) # Dynamic batch size, so needs to be set before running.
    z1 = np.random.uniform(-1, +1, [len(input_), 10])

    sprites = artist.run(y_op, feed_dict={
        z_ip: z1,
        l_ip: input_,
        b_size: batch_size
    })

    # Would be nice to combine these all into one sprite sheet.
    anim_one = createAnimation(0, 3, 6, sprites)
    anim_two = createAnimation(3, 3, 7, sprites)
    anim_three = createAnimation(6, 3, 8, sprites)
    anim_four = createAnimation(9, 3, 5, sprites)
    anim_five = createAnimation(12, 3, 12, sprites)
    anim_six = createAnimation(15, 1, 5, sprites)

    print("[Info] Sprites created successfully.")

    scipy.misc.imsave(app.root_path + "/static/images/sprite.png", joinImages(sprites))

    return jsonify(result=time.time())

if __name__ == '__main__':
    app.run()
