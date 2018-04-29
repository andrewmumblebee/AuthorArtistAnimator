# ANY UTILITY HELPERS, I.E Reading in files/Generating images.
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image, ImageFilter

def get_image_data(images, folder_path):
    """ Retrieves image data from a given path.

        Args:
            - images[list]: filenames to retrieve.
            - folder_path[string]: folder which contains the images.
    """
    data = []
    for file_path in images:
        im = Image.open(os.path.join(folder_path, file_path))
        data.append(np.reshape(im.getdata(), [64, 64, 4]))

    norm_data = ((np.array(data) / 255) * 2) - 1 # Normalize data
    return norm_data

def tileImage(imgs):
    """ Tiles images into a box, and denormalizes the values.

        Args:
            - imgs[array]: numpy array, containing images in form [?, 64, 64, 4]
    """
    d = int(np.sqrt(imgs.shape[0]-1))+1
    h, w = imgs.shape[1], imgs.shape[2]
    colour = imgs.shape[3]
    img = np.zeros((h * d, w * d, colour))
    for idx, image in enumerate(imgs):
        i = idx // d
        j = idx-i*d
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return ((img * 255.) + 1) * 2

class BatchGenerator:
    """ Class for handling retrieval of images from the dataset.

        Randomly samples from the dataset, until an epoch is completed then resets.
    """
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename.

            Args:
                - dataset_folder[string]: path to retrieve data from.
        """
        self.path, self.encoder = self.find_images(dataset_folder)
        self.reset_buffer()
        self.dataset_folder = dataset_folder

    def get_batch(self, batch_size):
        """ Retrieve a random batch of images with their base images and labels.

            Args:
                - batch_size[int]: size of batch to retrieve.
        """
        b_idx = np.random.randint(0, self.buffer.shape[0] - 1, batch_size) # Random index
        idx = self.buffer[b_idx]
        paths = self.path[idx]
        x = get_image_data(paths, self.dataset_folder) # Image and Respective Label
        l = self.get_encoding(paths)
        self.buffer = np.delete(self.buffer, b_idx)
        return x, l

    def get_labels(self, paths):
        """ Retrieve the labels encoded in the filename of the images.

            Args:
                - paths[list]: paths of the images to retrieve the labels from.
        """
        labels = []
        for path in paths:
            img_label = []
            ids = os.path.splitext(path)[0].split("_")
            for label in ids:
                if label.isdigit():
                    pass
                else:
                    img_label.append(label[1:])
            labels.append(img_label)
        return labels

    def get_encoding(self, paths):
        """ Apply hot encoding to labels before retrieving them.

            Args:
                - paths[list]: list of paths to retrieve labels for.
        """
        labels = self.get_labels(paths)
        encodings = self.encoder.transform(labels).toarray()
        return encodings

    def get_file_count(self):
        """ Retrieve amount of files in dataset. """
        return self.path.shape[0]

    def get_label_size(self):
        """ Retrieve size of encoded labels. """
        return sum(self.encoder.n_values_)

    def reset_buffer(self):
        """ Resets retrieval buffer, allowing a new epoch to retrieve the entire dataset again. """
        self.buffer = np.arange(self.path.shape[0])

    def generate_encoder(self, paths):
        """ Generates an encoder based on all labels in a dataset.

            Args:
                - paths[list]: files to retrieve labels and process encodings of.
        """
        labels = self.get_labels(paths)
        enc = OneHotEncoder()
        #labels = np.array(labels, dtype=np.float32)
        enc.fit(labels)
        labels = enc.transform(labels).toarray()
        return enc

    def find_images(self, path):
        """ Finds all the images within a given dataset folder, generating an encoder.

            Args:
                - paths[list]: files to retrieve labels and process encodings of.
        """
        paths = []
        for file in os.listdir(path):
            if not path.endswith('b.png'):
                paths.append(file)
        encoder = self.generate_encoder(paths)
        return np.array(paths), encoder


class AnimatorBatchGenerator(BatchGenerator):
    """ Class for handling retrieval of images from the dataset.

        Randomly samples from the dataset, until an epoch is completed then resets.
    """
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename. """
        self.path, self.encoder, self.base = self.find_images(dataset_folder)
        self.curr_frame = 0
        self.b_idx = 0
        self.reset_buffer()
        self.dataset_folder = dataset_folder

    # Need a second index in the buffer, corresponding to the frame to retrieve.

    def get_encoding(self, paths):
        """ Apply hot encoding to labels before retrieving them.

            Args:
                - paths[list]: list of paths to retrieve labels for.
        """
        animations, frames, _ = self.get_labels(paths)
        encodings = self.encoder.transform(animations).toarray()
        # Frames don't need to be hot encoded, so we just concatenate them to animation encodings.
        encodings = np.concatenate((encodings, frames), axis=1)
        return encodings

    def get_batch(self, batch_size):
        """ Retrieve a random batch of images with their base images and labels.

            Args:
                - batch_size[int]: size of batch to retrieve.
        """
        b_idx = np.random.randint(0, self.buffer.shape[0] - 1, batch_size) # Random index
        idx = self.buffer[b_idx]
        paths = self.path[idx]
        x = get_image_data(paths, self.dataset_folder) # Image and Respective Label
        l = self.get_encoding(paths)
        b = get_image_data(self.base[idx], self.dataset_folder)
        self.buffer = np.delete(self.buffer, b_idx)
        return x, l, b

    def generate_encoder(self, paths):
        """ Generates an encoder based on all labels in a dataset.

            Args:
                - paths[list]: files to retrieve labels and process encodings of.
        """
        animations, _, bases = self.get_labels(paths)
        enc = OneHotEncoder()
        enc.fit(animations)
        return enc, bases

    def get_label_size(self):
        """ Returns the overall length of the encoded labels. Plus 1 for frames. """
        return sum(self.encoder.n_values_) + 1

    def find_images(self, path):
        """ Finds all the images within a given dataset folder, generating an encoder.

            Args:
                - paths[list]: files to retrieve labels and process encodings of.
        """
        paths = []
        for file in os.listdir(path):
            if not file.endswith('b.png'):
                paths.append(file)
        paths = np.array(paths)
        encoder, bases = self.generate_encoder(paths)
        return paths, encoder, bases

    def get_labels(self, paths):
        """ Retrieve the labels encoded in the filename of the images.

            Args:
                - paths[list]: paths of the images to retrieve the labels from.
        """
        frames = []
        animations = []
        bases = []
        for path in paths:
            img_label = []
            ids = os.path.splitext(path)[0].split("_")
            animations.append([ids[0][1:]])
            frames.append([ids[1][1:]])
            bases.append(ids[0] + '_' + ids[2] + 'b.png')
        bases = np.array(bases)
        return animations, frames, bases
