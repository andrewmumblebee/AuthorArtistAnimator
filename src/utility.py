# ANY UTILITY HELPERS, I.E Reading in files/Generating images.
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image, ImageFilter

def get_image_data(images, path):
    data = []
    for file_path in images:
        im = Image.open(os.path.join(path, file_path))
        # x_dim = np.reshape(im, [-1, 64, 4])
        # copies = np.reshape(x_dim, [-1, 64, 64, 4])
        data.append(np.reshape(im.getdata(), [64, 64, 4]))
    norm_data = ((np.array(data) / 255)* 2) - 1
    return norm_data

def calcImageSize(dh, dw, stride):
    return int(np.ceil(float(dh)/float(stride))), int(np.ceil(float(dw)/float(stride)))

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
        return ((img * 255.) + 1) * 2
    else:
        raise ValueError('in merge(images,size) images parameter '
                        'must have dimensions: HxW or HxWx3 or HxWx4')

class BatchGenerator:
    """ Class for handling retrieval of images from the dataset.

        Randomly samples from the dataset, until an epoch is completed then resets.
    """
    def __init__(self, dataset_folder):
        """ Retrieve sprite images and label them based on their filename. """
        self.path, self.encoder = self.find_images(dataset_folder)
        self.reset_buffer()
        self.dataset_folder = dataset_folder

    def get_batch(self, batch_size, color=True):
        b_idx = np.random.randint(0, self.buffer.shape[0] - 1, batch_size) # Random index
        idx = self.buffer[b_idx]
        paths = self.path[idx]
        x = get_image_data(paths, self.dataset_folder) # Image and Respective Label
        l = self.get_encoding(paths)
        self.buffer = np.delete(self.buffer, b_idx)
        return x, l

    def get_labels(self, paths):
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
        labels = self.get_labels(paths)
        encodings = self.encoder.transform(labels).toarray()
        return encodings

    def get_file_count(self):
        return self.path.shape[0]

    def get_label_size(self):
        return sum(self.encoder.n_values_)

    def reset_buffer(self):
        self.buffer = np.arange(self.path.shape[0])

    def generate_encoder(self, paths):
        labels = self.get_labels(paths)
        enc = OneHotEncoder()
        #labels = np.array(labels, dtype=np.float32)
        enc.fit(labels)
        labels = enc.transform(labels).toarray()
        return enc

    def find_images(self, path):
        paths = []
        for file in os.listdir(path):
            if not path.endswith('b.png'):
                paths.append(file)
        encoder = self.generate_encoder(paths)
        # labels = labels / max_label_values
        # labels = (labels * 2) - 1 # Normalizing labels to -1 - 1 range. This assumes a minimum of 0 in original data.
        return np.array(paths), encoder


class ABatchGenerator(BatchGenerator):
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
        animations, frames, bases = self.get_labels(paths)
        encodings = self.encoder.transform(animations).toarray()
        encodings = np.concatenate((encodings, frames), axis=1)
        return encodings

    def get_batch(self, batch_size, color=True):
        b_idx = np.random.randint(0, self.buffer.shape[0] - 1, batch_size) # Random index
        idx = self.buffer[b_idx]
        paths = self.path[idx]
        x = get_image_data(paths, self.dataset_folder) # Image and Respective Label
        l = self.get_encoding(paths)
        b = get_image_data(self.base[idx], self.dataset_folder)
        self.buffer = np.delete(self.buffer, b_idx)
        return x, l, b

    def generate_encoder(self, paths):
        animations, _, bases = self.get_labels(paths)
        enc = OneHotEncoder()
        enc.fit(animations)
        return enc, bases

    def get_label_size(self):
        return sum(self.encoder.n_values_) + 1

    def find_images(self, path):
        paths = []
        for file in os.listdir(path):
            if not file.endswith('b.png'):
                paths.append(file)
        paths = np.array(paths)
        encoder, bases = self.generate_encoder(paths)
        # labels = labels / max_label_values
        # labels = (labels * 2) - 1 # Normalizing labels to -1 - 1 range. This assumes a minimum of 0 in original data.
        return paths, encoder, bases

    def get_labels(self, paths):
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


    # def get_batch(self, batch_size, color=True):
    #     batch_labels = np.array([])
    #     batch_images = np.array([])
    #     while batch_images.shape[0] < batch_size:

    #         curr_image = get_image_data([self.path[self.b_idx]], self.dataset_folder)
    #         print(curr_image.shape)

    #         if curr_image.shape[0] < self.curr_frame:
    #             batch_images.append(curr_image[self.curr_frame])
    #             batch_labels.append(curr_image[self.curr_frame])
    #             self.curr_frame += 1
    #         else:
    #             self.curr_frame = 0
    #             self.buffer = np.delete(self.buffer, self.b_idx)
    #             self.b_idx = self.buffer[np.random.randint(0, self.buffer.shape[0] - 1, 0)]

    #     idx = self.buffer[b_idx]
    #     x = get_image_data(self.path[idx], self.dataset_folder) # Image and Respective Label
    #     print(x.shape)
    #     x = (x / 255) # Normalize Channel values to 0-1 range.
    #     x = (x * 2) - 1 # Further Normalize to -1 to 1 range.
    #     t = self.label[idx]
    #     t[1] = self.curr_frame
    #     self.buffer = np.delete(self.buffer, b_idx)
    #     return batch_labels, batch_images


def main():
    img = get_image_data(['Female Dark 2 Blue Big Nose Big Ears.png'], '{}/../CharacterScraper/dump/sprites'.format(os.path.dirname(__file__)))
    paths, labels = find_images((os.path.join(os.path.dirname(__file__), '../CharacterScraper/dump/sprites')))

if __name__ == '__main__':
    main()